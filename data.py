import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted
import cv2
from typing import Dict, Any, List
import random, json
from plucker import ray_condition, RandomHorizontalFlipWithPose 

class StereoEventDataset(Dataset):
    def __init__(
        self, video_data_dir,
        frame_height=375, frame_width=375,
        random_seed=42):
        self.video_data_dir = video_data_dir
        np.random.seed(random_seed)

        self.frame_height = frame_height
        self.frame_width = frame_width
        video_names = sorted([v for v in os.listdir(video_data_dir)if os.path.isdir(os.path.join(video_data_dir, v))])
        self.video_names = video_names
        self.length = len(self.video_names)

        self.transform_rgb = transforms.Compose([transforms.Resize((frame_height, frame_width),interpolation=transforms.InterpolationMode.BILINEAR),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.transforms_evs = transforms.Compose([transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6, inplace=True)])
        self.pixel_transforms = [transforms.Resize((375, 375)),RandomHorizontalFlipWithPose(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.flip_flag = self.pixel_transforms[1].get_flip_flag(17)

    @staticmethod
    def great_filter(event_image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        event_image = event_image.astype(np.float32)
        max_val = np.max(event_image)
        if max_val > 0:
            event_image = event_image / max_val
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(event_image, kernel, iterations=iterations)
        eroded = cv2.erode(dilated, kernel, iterations=iterations)
        eroded = np.clip(eroded, 0, 1)
        return eroded

    @staticmethod
    def mask_function(event_image, kernel_size=31, kernel_size_erode=61,kernel_size_midele=31, iterations=1, sigma_log=10):
        max_value = np.max(np.abs(event_image))
        if max_value != 0:
            event_image = np.abs(event_image) / max_value
        else:
            event_image = np.abs(event_image)

        event_image_blurred = cv2.GaussianBlur(event_image, (kernel_size,kernel_size), sigma_log)
        _, binary_image = cv2.threshold(event_image_blurred, 0.01, 1, cv2.THRESH_BINARY)    
        kernel_dilate = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)
        binary_image_dilated = cv2.dilate(binary_image, kernel_dilate, iterations=iterations )
        binary_median = cv2.medianBlur(binary_image_dilated.astype(np.uint8), kernel_size_midele)
        return binary_median

    def _load_image(self, image_path: str, channels: int = 3):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)    
        img = img.astype(np.float32) / 255.0
        return img
    
    def get_video_paths(self, video_names):  
        paths = {}
        for cam, folder in zip(['left', 'right'], ['camera_00_rgb_depth_events', 'camera_01_rgb_depth_events']):
            cam_path = os.path.join(self.video_data_dir, video_names, folder)
            rgb = natsorted(glob(os.path.join(cam_path, 'images', '*.png')))
            depth = natsorted(glob(os.path.join(cam_path, 'depths', '*.npy')))
            event = natsorted(glob(os.path.join(cam_path, 'events', '*.png')))
            meta_file_path = os.path.join(cam_path, 'metadata.json')
            camera_metadata = {}
            with open(meta_file_path, 'r') as f:
                camera_metadata = json.load(f)
            paths[cam] = {'rgb': rgb,'depth': depth, 'event': event, 'metadata': camera_metadata,'metadata_path': meta_file_path}
        return paths

    def _get_paths(self, video_names):
        return self.get_video_paths(video_names)

    def __len__(self):
        return self.length

    def _load_rgb(self, rgb_paths: List[str]):
        frames = [self._load_image(path, channels=3) for path in rgb_paths]
        self.num_rgb_frames = len(frames)
        rgb_stack = np.stack(frames) 
        rgb_stack = np.transpose(rgb_stack, (0, 3, 1, 2))
        rgb_stack = torch.from_numpy(rgb_stack).float()
        rgb_stack = (rgb_stack - torch.min(rgb_stack))/( torch.max(rgb_stack) - torch.min(rgb_stack) )
        return rgb_stack

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        depth_list = []
        for path in depth_paths:
            depth_map = np.load(path).astype(np.float32)
            depth_resized = cv2.resize(depth_map, (self.frame_width, self.frame_height),interpolation=cv2.INTER_NEAREST)
            depth_list.append(depth_resized)
        depth_stack = np.stack(depth_list)
        depth_stack = np.expand_dims(depth_stack, axis=1) 
        return torch.from_numpy(depth_stack).float()

    def _load_events(self, event_paths, num_timesteps= 16, num_channels= 6):
        expected_files = num_timesteps * num_channels 
        event_frames = []
        for t in range(num_timesteps): 
            channels = []
            for ch in range(num_channels):
                idx = t * num_channels + ch 
                
                if idx < len(event_paths):
                    img = cv2.imread(event_paths[idx], cv2.IMREAD_UNCHANGED) 
                    filtered = StereoEventDataset.great_filter(img) 
                    channels.append(filtered)
                else:
                    channels.append(np.zeros((self.frame_height, self.frame_width), dtype=np.float32))
                    
            frame = np.stack(channels, axis=-1)
            event_frames.append(frame)

        event_stack = np.stack(event_frames)
        event_stack = np.transpose(event_stack, (0, 3, 1, 2))
        event_tensor = torch.from_numpy(event_stack).float()
        
        blank_frame = torch.zeros((1, num_channels, self.frame_height, self.frame_width),dtype=torch.float32)
        event_tensor = torch.cat([blank_frame, event_tensor], dim=0)
        return event_tensor


    def crop_center_patch(self, pixel_values, event_voxel_bin, crop_h=375, crop_w=375, random_crop=False):
        height = pixel_values.shape[2]
        width = pixel_values.shape[3]
        if random_crop:
            start_h = random.randint(0, height - crop_h)
            start_w = random.randint(0, width - crop_w)
        else:
            center_h = height // 2
            center_w = width // 2
            start_h = center_h - crop_h // 2
            start_w = center_w - crop_w // 2
        cropped_pixel_values = pixel_values[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped_event_voxel_bin = event_voxel_bin[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        return cropped_pixel_values, cropped_event_voxel_bin


    def plucker_embeddings(self, paths, flip_flag, frame_height=375, frame_width=375):
        EXTRINSICS_KEY = 'extrinsics'
        INTRINSICS_KEY = 'intrinsics'
        embeddings = {}
        for cam in ['left', 'right']:
            camera_paras = paths[cam]['metadata']
            extrinsics_np = np.array(camera_paras[EXTRINSICS_KEY], dtype=np.float32)
            c2w_tensor = torch.as_tensor(extrinsics_np).unsqueeze(0)  
            K_matrix_np = np.array(camera_paras[INTRINSICS_KEY], dtype=np.float32)
            intrinsics_vec_np = np.array([K_matrix_np[0,0], K_matrix_np[1,1], K_matrix_np[0,2], K_matrix_np[1,2]], dtype=np.float32)
            intrinsics_tensor = torch.as_tensor(intrinsics_vec_np).unsqueeze(0).repeat(1, 17, 1) 
            embedding = ray_condition(intrinsics_tensor, c2w_tensor, frame_height, frame_width, device='cpu', flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
            embeddings[cam] = embedding
        return embeddings['left'], embeddings['right']

    def __getitem__(self, idx: int):
        video_name = self.video_names[idx]
        paths = self._get_paths(video_name)
        left_meta_path = paths['left']['metadata_path']
        right_meta_path = paths['right']['metadata_path']
        left_rgb = self._load_rgb(paths['left']['rgb'])
        left_depth = self._load_depth(paths['left']['depth'])
        left_event = self._load_events(paths['left']['event'])

        right_rgb = self._load_rgb(paths['right']['rgb'])
        right_depth = self._load_depth(paths['right']['depth'])
        right_event = self._load_events(paths['right']['event'])

        plucker_embedding_left, plucker_embedding_right = self.plucker_embeddings(paths=paths,flip_flag=self.flip_flag,frame_height=self.frame_height,frame_width=self.frame_width)
        def apply_transform_to_sequence(sequence_tensor, transform_fn):
            if sequence_tensor.ndim == 3:
                return transform_fn(sequence_tensor)
            transformed_frames = []
            for t in range(sequence_tensor.shape[0]):
                transformed_frame = transform_fn(sequence_tensor[t])
                transformed_frames.append(transformed_frame)
            return torch.stack(transformed_frames, dim=0)

        left_rgb = apply_transform_to_sequence(left_rgb, self.transform_rgb)
        right_rgb = apply_transform_to_sequence(right_rgb, self.transform_rgb)

        left_event = apply_transform_to_sequence(left_event, self.transforms_evs)
        right_event = apply_transform_to_sequence(right_event, self.transforms_evs)

        left_pixel_values, left_events = self.crop_center_patch(pixel_values=left_rgb,event_voxel_bin=left_event,random_crop=True)
        right_pixel_values, right_events = self.crop_center_patch(pixel_values=right_rgb,event_voxel_bin=right_event,random_crop=True)

        final_frame = left_pixel_values[-1] 
        sample = dict(left=dict(pixel_values=left_pixel_values,events=left_events,depth=left_depth,plucker_embedding=plucker_embedding_left),
            right=dict(pixel_values=right_pixel_values,events=right_events,depth=right_depth,plucker_embedding=plucker_embedding_right), video_name=video_name)
        return sample

# for batch_idx, batch in enumerate(train_dataloader):
#     video_name = batch['video_name'][0]     
#     print(f"================== Processing Batch {batch_idx + 1} ==================")
#     print(f"Current Video: {video_name}")
#     left_pixel_values_batch = batch['left']['pixel_values'] 
#     left_events_batch = batch['left']['events']             
#     left_depth_batch = batch['left']['depth']               
#     left_plucker_embedding_batch = batch['left']['plucker_embedding'] 

#     right_pixel_values_batch = batch['right']['pixel_values']
#     right_events_batch = batch['right']['events']
#     right_depth_batch = batch['right']['depth']
#     right_plucker_embedding_batch = batch['right']['plucker_embedding'] 
#     print(right_events_batch.shape)
