import argparse
from pathlib import Path

from src.configs.train_config import TrainConfig, LogConfig, RenderConfig, GuideConfig, OptimConfig
from src.training.trainer import TEXTure


def main():
    parser = argparse.ArgumentParser(description='Run TEXTure with direct parameters')
    
    # Required parameters
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to your .obj mesh file')
    parser.add_argument('--text_prompt', type=str, required=True, help='Text prompt for texture generation')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name for output directory')
    
    # Optional parameters with defaults
    parser.add_argument('--shape_scale', type=float, default=0.6, help='Scale of mesh in 1x1x1 cube (default: 0.6)')
    parser.add_argument('--texture_resolution', type=int, default=1024, help='Texture resolution (default: 1024)')
    parser.add_argument('--guidance_scale', type=float, default=10.0, help='Guidance scale for diffusion (default: 10.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--diffusion_model', type=str, default='stabilityai/stable-diffusion-2-depth', 
                       help='Diffusion model to use (default: stabilityai/stable-diffusion-2-depth)')
    parser.add_argument('--append_direction', action='store_true', default=True, 
                       help='Append view direction to text prompt (default: True)')
    parser.add_argument('--eval_only', action='store_true', default=False, 
                       help='Run only evaluation (default: False)')
    
    args = parser.parse_args()
    
    # Create configuration objects
    log_config = LogConfig(
        exp_name=args.exp_name,
        exp_root=Path('experiments/'),
        eval_only=args.eval_only,
        eval_size=10,
        full_eval_size=100,
        save_mesh=True,
        vis_diffusion_steps=False,
        log_images=True
    )
    
    render_config = RenderConfig(
        train_grid_size=1200,
        eval_grid_size=1024,
        radius=1.5,
        overhead_range=40,
        front_range=70,
        front_offset=0.0,
        n_views=8,
        base_theta=60,
        views_before=[],
        views_after=[[180, 30], [180, 150]],
        alternate_views=True
    )
    
    guide_config = GuideConfig(
        text=args.text_prompt,
        shape_path=args.mesh_path,
        append_direction=args.append_direction,
        concept_name=None,
        concept_path=None,
        diffusion_name=args.diffusion_model,
        shape_scale=args.shape_scale,
        dy=0.25,
        texture_resolution=args.texture_resolution,
        texture_interpolation_mode='bilinear',
        guidance_scale=args.guidance_scale,
        use_inpainting=True,
        reference_texture=None,
        initial_texture=None,
        use_background_color=False,
        background_img='textures/brick_wall.png',
        z_update_thr=0.2,
        strict_projection=True
    )
    
    optim_config = OptimConfig(
        seed=args.seed,
        lr=1e-2,
        min_timestep=0.02,
        max_timestep=0.98,
        no_noise=False
    )
    
    # Create main config
    config = TrainConfig(
        log=log_config,
        render=render_config,
        optim=optim_config,
        guide=guide_config
    )
    
    # Run TEXTure
    trainer = TEXTure(config)
    if config.log.eval_only:
        trainer.full_eval()
    else:
        trainer.paint()


if __name__ == '__main__':
    main() 