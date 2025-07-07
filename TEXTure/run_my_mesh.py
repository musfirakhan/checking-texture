#!/usr/bin/env python3
"""
Simple script to run TEXTure with direct parameters
Usage: python run_my_mesh.py
"""

from pathlib import Path
from src.configs.train_config import TrainConfig, LogConfig, RenderConfig, GuideConfig, OptimConfig
from src.training.trainer import TEXTure


def run_texture_for_mesh(
    mesh_path: str,
    text_prompt: str,
    exp_name: str,
    shape_scale: float = 0.6,
    texture_resolution: int = 1024,
    guidance_scale: float = 10.0,
    seed: int = 42,
    diffusion_model: str = 'stabilityai/stable-diffusion-2-depth'
):
    """
    Run TEXTure with your custom parameters
    
    Args:
        mesh_path: Path to your .obj mesh file
        text_prompt: Text prompt for texture generation (use {} for view direction)
        exp_name: Name for the experiment output directory
        shape_scale: Scale of mesh in 1x1x1 cube (default: 0.6)
        texture_resolution: Texture resolution (default: 1024)
        guidance_scale: Guidance scale for diffusion (default: 10.0)
        seed: Random seed (default: 42)
        diffusion_model: Diffusion model to use
    """
    
    # Create configuration objects
    log_config = LogConfig(
        exp_name=exp_name,
        exp_root=Path('experiments/'),
        eval_only=False,
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
        text=text_prompt,
        shape_path=mesh_path,
        append_direction=True,
        concept_name=None,
        concept_path=None,
        diffusion_name=diffusion_model,
        shape_scale=shape_scale,
        dy=0.25,
        texture_resolution=texture_resolution,
        texture_interpolation_mode='bilinear',
        guidance_scale=guidance_scale,
        use_inpainting=True,
        reference_texture=None,
        initial_texture=None,
        use_background_color=False,
        background_img='textures/brick_wall.png',
        z_update_thr=0.2,
        strict_projection=True
    )
    
    optim_config = OptimConfig(
        seed=seed,
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
    trainer.paint()
    
    print(f"✅ TEXTure completed! Results saved in experiments/{exp_name}/")


if __name__ == '__main__':
    # ===== EDIT THESE PARAMETERS FOR YOUR MESH =====
    
    # Required parameters
    MESH_PATH = "shapes/your_mesh.obj"  # Path to your .obj file
    TEXT_PROMPT = "A beautiful red car, {} view"  # Text prompt ({} will be replaced with view direction)
    EXP_NAME = "my_car_experiment"  # Name for output directory
    
    # Optional parameters (you can adjust these)
    SHAPE_SCALE = 0.6  # Scale of your mesh (0.6 = 60% of 1x1x1 cube)
    TEXTURE_RESOLUTION = 1024  # Texture resolution
    GUIDANCE_SCALE = 10.0  # How closely to follow text prompt (higher = more faithful)
    SEED = 42  # Random seed for reproducible results
    DIFFUSION_MODEL = 'stabilityai/stable-diffusion-2-depth'  # Diffusion model
    
    # ===== RUN TEXTURE =====
    run_texture_for_mesh(
        mesh_path=MESH_PATH,
        text_prompt=TEXT_PROMPT,
        exp_name=EXP_NAME,
        shape_scale=SHAPE_SCALE,
        texture_resolution=TEXTURE_RESOLUTION,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
        diffusion_model=DIFFUSION_MODEL
    ) 