#!/usr/bin/env python3

import torch
import numpy as np
import time
import os
import pickle
import argparse
import cv2
import json
import zmq
from types import SimpleNamespace
from torchvision import transforms

from detr.models.detr_vae import build as build_ACT_model
from detr.models.detr_vae import build_cnnmlp as build_CNNMLP_model
from detr.models.backbone import build_backbone
from detr.models.transformer import build_transformer

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class ACTPolicyDeploy(torch.nn.Module):
    """ACT Policy wrapper for deployment"""
    def __init__(self, model, kl_weight):
        super().__init__()
        self.model = model
        self.kl_weight = kl_weight
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    
    def forward(self, qpos, image):
        """Inference only forward pass"""
        image = self.normalize(image)
        env_state = None
        a_hat, _, _, _, _ = self.model(qpos, image, env_state, vq_sample=None)
        return a_hat


class D1ACTController:
    def __init__(self, ckpt_path, temporal_agg=True):
        # Load model
        self._load_model(ckpt_path)
        self.temporal_agg = temporal_agg
        
        # Robot control
        self.sport_client = None
        self.msc = None
        
        # ZMQ setup
        self.zmq_context = zmq.Context()
        self.d1_cmd_socket = None
        self.d1_state_sub = None
        self.front_cam_sub = None
        self.wrist_cam_sub = None
        
        # Control parameters
        self.control_freq = 50  # Hz
        self.dt = 1.0 / self.control_freq
        self.chunk_size = self.config['policy_config']['num_queries']
        
        # State
        self.current_qpos = np.zeros(7)
        self.current_images = {}
        self.step_count = 0
        
        # Temporal aggregation
        if temporal_agg:
            self.all_time_actions = torch.zeros([
                1000, 1000 + self.chunk_size, self.action_dim
            ]).cuda()
    
    def _load_model(self, ckpt_path):
        """Load ACT model without using build_ACT_model_and_optimizer"""
        print(f"Loading model from {ckpt_path}")
        
        # Load config
        ckpt_dir = os.path.dirname(ckpt_path)
        with open(os.path.join(ckpt_dir, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)
        
        # Load stats
        with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
            self.stats = pickle.load(f)
        
        # Model dimensions
        self.state_dim = self.config['state_dim']  # 7
        self.action_dim = self.config['policy_config']['action_dim']  # 10
        self.camera_names = self.config['camera_names']
        
        # Create policy based on class
        policy_class = self.config['policy_class']
        policy_config = self.config['policy_config']
        
        if policy_class == 'ACT':
            args = SimpleNamespace()
            
            for k, v in policy_config.items():
                setattr(args, k, v)
            
            args.state_dim = self.state_dim
            args.dropout = 0.1
            args.pre_norm = False
            args.dilation = False
            args.position_embedding = 'sine'
            args.masks = False
            args.weight_decay = 1e-4
            args.lr_drop = 200
            args.clip_max_norm = 0.1
            args.epochs = 300
            args.batch_size = 2
            
            if not hasattr(args, 'lr'):
                args.lr = 1e-4
            if not hasattr(args, 'lr_backbone'):
                args.lr_backbone = 1e-5
            if not hasattr(args, 'backbone'):
                args.backbone = 'resnet18'
            if not hasattr(args, 'enc_layers'):
                args.enc_layers = 4
            if not hasattr(args, 'dec_layers'):
                args.dec_layers = 7
            if not hasattr(args, 'dim_feedforward'):
                args.dim_feedforward = 2048
            if not hasattr(args, 'hidden_dim'):
                args.hidden_dim = 256
            if not hasattr(args, 'nheads'):
                args.nheads = 8
            if not hasattr(args, 'num_queries'):
                args.num_queries = 100
            if not hasattr(args, 'vq'):
                args.vq = False
            if not hasattr(args, 'vq_class'):
                args.vq_class = 0
            if not hasattr(args, 'vq_dim'):
                args.vq_dim = 0
            if not hasattr(args, 'no_encoder'):
                args.no_encoder = False
            
            # Build the model
            print("Building ACT model...")
            model = build_ACT_model(args)
            
            # Create policy wrapper
            self.policy = ACTPolicyDeploy(model, policy_config.get('kl_weight', 10))
            
        else:
            raise ValueError(f"Unsupported policy class: {policy_class}")
        
        # Load weights
        print("Loading weights...")
        state_dict = torch.load(ckpt_path)
        
        if 'model' in state_dict:
            self.policy.load_state_dict(state_dict)
        else:
            try:
                self.policy.model.load_state_dict(state_dict)
            except:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k] = v
                    else:
                        new_state_dict[f'model.{k}'] = v
                self.policy.load_state_dict(new_state_dict)
        
        self.policy.cuda()
        self.policy.eval()
        
        print(f"Model loaded successfully (Policy class: {policy_class})")
    
    def connect_robot(self):
        """Connect to Go2 robot"""
        ChannelFactoryInitialize(0, "enp14s0")
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        print("Go2 Robot connected")
        
        self.msc = MotionSwitcherClient()
        self.msc.SelectMode("ai")
        print("AI motion mode enabled")
    
    def init_zmq(self):
        """Initialize ZMQ connections"""
        # D1 command publisher
        self.d1_cmd_socket = self.zmq_context.socket(zmq.PUB)
        self.d1_cmd_socket.bind("tcp://*:5555")
        
        # Subscribers
        self.d1_state_sub = self.zmq_context.socket(zmq.SUB)
        self.d1_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.d1_state_sub.setsockopt(zmq.CONFLATE, 1)
        self.d1_state_sub.connect("tcp://localhost:5556")
        
        self.front_cam_sub = self.zmq_context.socket(zmq.SUB)
        self.front_cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.front_cam_sub.setsockopt(zmq.CONFLATE, 1)
        self.front_cam_sub.connect("tcp://192.168.123.18:5557")
        
        self.wrist_cam_sub = self.zmq_context.socket(zmq.SUB)
        self.wrist_cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.wrist_cam_sub.setsockopt(zmq.CONFLATE, 1)
        self.wrist_cam_sub.connect("tcp://192.168.123.18:5558")
        
        time.sleep(0.5)
        print("ZMQ connections initialized")
    
    def get_observations(self):
        """Get current robot state and camera images"""
        # Get D1 joint positions
        try:
            msg = self.d1_state_sub.recv_string(zmq.NOBLOCK)
            data = json.loads(msg)
            self.current_qpos = np.array(data['joint_positions'], dtype=np.float32)
        except:
            pass
        
        # Get camera images
        try:
            jpg_buffer = self.front_cam_sub.recv(zmq.NOBLOCK)
            img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.current_images['front_image'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass
        
        try:
            jpg_buffer = self.wrist_cam_sub.recv(zmq.NOBLOCK)
            img_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.current_images['wrist_image'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass
    
    def preprocess_inputs(self):
        """Prepare inputs for policy"""
        # Normalize qpos
        qpos_normalized = (self.current_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        qpos_tensor = torch.from_numpy(qpos_normalized).float().cuda().unsqueeze(0)
        
        # Process images
        images = []
        for cam_name in self.camera_names:
            if cam_name in self.current_images:
                img = self.current_images[cam_name]
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                images.append(img_tensor)
            else:
                # Black image if camera not ready
                print(f"Warning: {cam_name} not available, using black image")
                images.append(torch.zeros(3, 480, 640))
        
        images = torch.stack(images, dim=0).unsqueeze(0).cuda()
        
        return qpos_tensor, images
    
    def get_action(self, qpos_tensor, images):
        """Get action from policy"""
        with torch.no_grad():
            if self.temporal_agg:
                # Query policy
                all_actions = self.policy(qpos_tensor, images)
                
                # Temporal aggregation
                t = self.step_count
                self.all_time_actions[[t], t:t+self.chunk_size] = all_actions
                actions_for_curr_step = self.all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                if len(actions_for_curr_step) > 0:
                    # Exponential weighting
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).float().cuda().unsqueeze(dim=1)
                    
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    # First step
                    raw_action = all_actions[:, 0]
            else:
                # Simple query
                all_actions = self.policy(qpos_tensor, images)
                raw_action = all_actions[:, self.step_count % self.chunk_size]
        
        # Denormalize
        action = raw_action.squeeze(0).cpu().numpy()
        action = action * self.stats['action_std'] + self.stats['action_mean']
        
        return action
    
    def send_commands(self, action):
        """Send commands to robot"""
        # Split action
        base_vel = action[:3]  # [x, y, yaw]
        d1_pos = action[3:10]  # 7 joint positions
        
        # Convert numpy float32 to Python float for JSON serialization
        self.sport_client.Move(
            float(base_vel[1]),  # y velocity
            float(base_vel[0]),  # x velocity
            float(base_vel[2])   # yaw velocity
        )
        
        # Send D1 joint positions
        d1_cmd = {
            "positions": [float(pos) for pos in d1_pos],  # Convert to Python float
            "timestamp": time.time()
        }
        self.d1_cmd_socket.send_json(d1_cmd, zmq.NOBLOCK)
    
    def run(self):
        """Main control loop"""
        print("\nStarting ACT control...")
        print("Warming up...")
        
        # Wait for initial observations
        for _ in range(10):
            self.get_observations()
            time.sleep(0.1)
        
        # Check if we have all required data
        if len(self.current_images) < len(self.camera_names):
            print("Warning: Not all cameras are available")
            print(f"Required: {self.camera_names}")
            print(f"Available: {list(self.current_images.keys())}")
        
        # Warm up model
        qpos_tensor, images = self.preprocess_inputs()
        for _ in range(10):
            _ = self.policy(qpos_tensor, images)
        
        print("Control active! Press Ctrl+C to stop")
        
        # Main loop
        try:
            while True:
                start_time = time.time()
                
                # Get observations
                self.get_observations()
                
                # Get action from policy
                qpos_tensor, images = self.preprocess_inputs()
                action = self.get_action(qpos_tensor, images)
                
                # Send commands
                self.send_commands(action)
                
                # Timing
                self.step_count += 1
                elapsed = time.time() - start_time
                time.sleep(max(0, self.dt - elapsed))
                
                # Status
                if self.step_count % 50 == 0:
                    print(f"Step {self.step_count}, Control rate: {1/elapsed:.1f} Hz")
                
        except KeyboardInterrupt:
            print("\nStopping...")
    
    def cleanup(self):
        """Clean up resources"""
        # Stop robot
        if self.sport_client:
            self.sport_client.Move(0, 0, 0)
            self.sport_client.BalanceStand()
        
        # Send zero commands to D1
        if self.d1_cmd_socket:
            zero_cmd = {
                "positions": [0.0] * 7,
                "timestamp": time.time()
            }
            self.d1_cmd_socket.send_json(zero_cmd)
        
        # Close ZMQ
        if self.d1_cmd_socket:
            self.d1_cmd_socket.close()
        if self.d1_state_sub:
            self.d1_state_sub.close()
        if self.front_cam_sub:
            self.front_cam_sub.close()
        if self.wrist_cam_sub:
            self.wrist_cam_sub.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--no-temporal-agg', action='store_true', help='Disable temporal aggregation')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found: {args.ckpt}")
        return
    
    # Create controller
    controller = D1ACTController(
        ckpt_path=args.ckpt,
        temporal_agg=not args.no_temporal_agg
    )
    
    try:
        # Initialize connections
        controller.connect_robot()
        controller.init_zmq()
        
        # Run control
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()