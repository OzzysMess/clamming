"""
N-Mass-Spring-Damper System with Friction
A modular implementation for simulating and visualizing multi-mass systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import imageio.v2 as iio
import time as time_module


class MassSpringDamperSystem:
    """
    Simulates an n-mass spring-damper system with friction.
    
    System Configuration:
    - N masses connected in series by springs and dampers
    - Last mass connected to fixed wall via spring and damper
    - Input force on first mass: F(t) = A·sin(ωt)
    - Coulomb friction (smooth approximation) on all masses
    """
    
    def __init__(self, n=10, m=None, k=None, c=None, mu_friction=0.25,
                 omega=2.0, amplitude=10.0, t_end=10.0, num_points=10000):
        """
        Initialize the system.
        
        Parameters:
        -----------
        n : int
            Number of masses
        m : array-like or float, optional
            Mass values (default: 1.0 kg for all)
        k : array-like or float, optional
            Spring constants (default: 10.0 N/m for all)
        c : array-like or float, optional
            Damping coefficients (default: 0.5 N·s/m for all)
        mu_friction : float
            Friction coefficient (default: 0.25)
        omega : float
            Input force frequency in rad/s (default: 2.0)
        amplitude : float
            Input force amplitude in N (default: 10.0)
        t_end : float
            Simulation end time in seconds (default: 10.0)
        num_points : int
            Number of time points for simulation (default: 10000)
        """
        self.n = n
        self.m = np.ones(n) * m if isinstance(m, (int, float)) else (m if m is not None else np.ones(n))
        self.k = np.ones(n) * k if isinstance(k, (int, float)) else (k if k is not None else np.ones(n) * 10.0)
        self.c = np.ones(n) * c if isinstance(c, (int, float)) else (c if c is not None else np.ones(n) * 0.5)
        self.mu_friction = mu_friction
        self.omega = omega
        self.amplitude = amplitude
        self.t_end = t_end
        self.num_points = num_points
        
        # Time vector
        self.t = np.linspace(0, t_end, num_points)
        
        # Solution storage
        self.solution = None
        self.positions = None
        self.velocities = None
        self.F_input = None
        
        self._print_config()
    
    def _print_config(self):
        """Print system configuration."""
        print(f"Mass-Spring-Damper System Configuration:")
        print(f"  Number of masses: {self.n}")
        print(f"  Masses: {self.m[0]} kg (uniform)" if np.allclose(self.m, self.m[0]) else f"  Masses: {self.m}")
        print(f"  Spring constants: {self.k[0]} N/m (uniform)" if np.allclose(self.k, self.k[0]) else f"  Spring constants: {self.k}")
        print(f"  Damping coefficients: {self.c[0]} N·s/m (uniform)" if np.allclose(self.c, self.c[0]) else f"  Damping coefficients: {self.c}")
        print(f"  Friction coefficient (μ): {self.mu_friction}")
        print(f"  Input force: F(t) = {self.amplitude}·sin({self.omega}·t) N")
        print(f"  Simulation time: {self.t_end} seconds\n")
    
    def _dynamics(self, x, t):
        """
        System dynamics: dx/dt = f(x, t)
        
        State vector: x = [x1, x2, ..., xn, v1, v2, ..., vn]
        """
        positions = x[:self.n]
        velocities = x[self.n:]
        
        # Input force
        F_input = self.amplitude * np.sin(self.omega * t)
        
        # Smooth friction using tanh
        g = 9.81
        v_smooth = 0.01
        F_friction = -self.mu_friction * self.m * g * np.tanh(velocities / v_smooth)
        
        accelerations = np.zeros(self.n)
        
        # First mass
        if self.n > 1:
            F_spring = (self.k[0] * (positions[0] - positions[1]) + 
                       self.c[0] * (velocities[0] - velocities[1]))
            accelerations[0] = (F_input - F_spring + F_friction[0]) / self.m[0]
        else:
            F_spring = self.k[0] * positions[0] + self.c[0] * velocities[0]
            accelerations[0] = (F_input - F_spring + F_friction[0]) / self.m[0]
        
        # Middle masses
        for i in range(1, self.n - 1):
            F_left = (self.k[i-1] * (positions[i-1] - positions[i]) + 
                     self.c[i-1] * (velocities[i-1] - velocities[i]))
            F_right = (self.k[i] * (positions[i] - positions[i+1]) + 
                      self.c[i] * (velocities[i] - velocities[i+1]))
            accelerations[i] = (F_left - F_right + F_friction[i]) / self.m[i]
        
        # Last mass (connected to wall)
        if self.n > 1:
            F_left = (self.k[self.n-2] * (positions[self.n-2] - positions[self.n-1]) + 
                     self.c[self.n-2] * (velocities[self.n-2] - velocities[self.n-1]))
            F_wall = self.k[self.n-1] * positions[self.n-1] + self.c[self.n-1] * velocities[self.n-1]
            accelerations[self.n-1] = (F_left - F_wall + F_friction[self.n-1]) / self.m[self.n-1]
        
        return np.concatenate([velocities, accelerations])
    
    def solve(self):
        """
        Solve the ODE system.
        
        Returns:
        --------
        positions : ndarray, shape (num_points, n)
            Position of each mass over time
        velocities : ndarray, shape (num_points, n)
            Velocity of each mass over time
        """
        x0 = np.zeros(2 * self.n)
        self.solution = odeint(self._dynamics, x0, self.t)
        self.positions = self.solution[:, :self.n]
        self.velocities = self.solution[:, self.n:]
        self.F_input = self.amplitude * np.sin(self.omega * self.t)
        
        print(f"Simulation completed successfully!")
        print(f"  Final positions (m): {self.positions[-1]}")
        print(f"  Final velocities (m/s): {self.velocities[-1]}\n")
        
        return self.positions, self.velocities
    
    def plot_trajectories(self):
        """Plot positions, velocities, and input force."""
        assert self.positions is not None, "Must call solve() first"
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Positions
        for i in range(self.n):
            axes[0].plot(self.t, self.positions[:, i], 'k-', linewidth=1.5, alpha=0.7, label=f'M{i+1}')
        axes[0].set_ylabel('Position (m)', fontsize=11)
        axes[0].set_title(f'{self.n}-Mass System: Positions vs Time', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper right', ncol=min(self.n, 5))
        axes[0].grid(True, alpha=0.3)
        
        # Velocities
        for i in range(self.n):
            axes[1].plot(self.t, self.velocities[:, i], 'k-', linewidth=1.5, alpha=0.7, label=f'M{i+1}')
        axes[1].set_xlabel('Time (s)', fontsize=11)
        axes[1].set_ylabel('Velocity (m/s)', fontsize=11)
        axes[1].set_title(f'{self.n}-Mass System: Velocities vs Time', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', ncol=min(self.n, 5))
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Input force
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.t, self.F_input, 'k-', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Force (N)', fontsize=11)
        ax.set_title('Input Force: F(t) = sin(ωt)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_phase_portraits(self):
        """Plot phase portraits (position vs velocity) for each mass."""
        assert self.positions is not None, "Must call solve() first"
        
        n_cols = min(3, self.n)
        n_rows = (self.n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i in range(self.n):
            ax = axes[i]
            ax.plot(self.positions[:, i], self.velocities[:, i], 'k-', linewidth=2)
            ax.scatter(self.positions[0, i], self.velocities[0, i], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(self.positions[-1, i], self.velocities[-1, i], c='red', s=100, marker='s', label='End', zorder=5)
            ax.set_xlabel('Position (m)', fontsize=10)
            ax.set_ylabel('Velocity (m/s)', fontsize=10)
            ax.set_title(f'Phase Portrait: Mass {i+1}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        for j in range(self.n, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_energy_momentum(self):
        """Plot total kinetic energy and momentum vs time."""
        assert self.positions is not None, "Must call solve() first"
        
        # Kinetic energy
        KE_individual = 0.5 * self.m[np.newaxis, :] * self.velocities**2
        KE_total = np.sum(KE_individual, axis=1)
        
        # Momentum
        momentum_individual = self.m[np.newaxis, :] * self.velocities
        momentum_total = np.sum(momentum_individual, axis=1)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Kinetic energy
        axes[0].plot(self.t, KE_total, 'k-', linewidth=2)
        axes[0].fill_between(self.t, KE_total, alpha=0.3, color='black')
        axes[0].set_ylabel('Total Kinetic Energy (J)', fontsize=11)
        axes[0].set_title('System Total Kinetic Energy vs Time', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Momentum
        axes[1].plot(self.t, momentum_total, 'k-', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].fill_between(self.t, momentum_total, alpha=0.3, color='black')
        axes[1].set_xlabel('Time (s)', fontsize=11)
        axes[1].set_ylabel('Total Momentum (kg·m/s)', fontsize=11)
        axes[1].set_title('System Total Momentum vs Time', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Energy and Momentum Statistics:")
        print(f"  Maximum kinetic energy: {np.max(KE_total):.4f} J")
        print(f"  Final kinetic energy: {KE_total[-1]:.4f} J")
        print(f"  Maximum total momentum: {np.max(np.abs(momentum_total)):.4f} kg·m/s")
        print(f"  Final total momentum: {momentum_total[-1]:.4f} kg·m/s\n")
    
    def animate(self, fps=20, output_file=None):
        """
        Generate real-time animation video.
        
        Parameters:
        -----------
        fps : int
            Frames per second (default: 20)
        output_file : str, optional
            Path to save video file. If None, saves to temp directory.
        
        Returns:
        --------
        video_path : str
            Path to the generated video file
        """
        assert self.positions is not None, "Must call solve() first"
        
        # Setup
        num_frames = int(fps * self.t_end)
        stride = len(self.t) / num_frames
        frame_indices = [int(i * stride) for i in range(num_frames)]
        
        if output_file is None:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = os.path.dirname(output_file) or '.'
        
        print(f"Generating animation ({num_frames} frames at {fps} fps)...")
        
        # Image parameters
        img_width, img_height = 1200, 300
        y_center = img_height // 2
        mass_radius = 10
        
        def x_to_pixel(x_val):
            return int((x_val + 0.8) / 2.0 * (img_width - 100) + 50)
        
        # Generate frames
        start_time = time_module.time()
        for frame_num, frame_idx in enumerate(frame_indices):
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Centerline
            draw.line([(0, y_center), (img_width, y_center)], fill='#eeeeee', width=1)
            
            # Wall
            wall_x_pix = x_to_pixel(0.95)
            for wall_y_offset in range(-60, 60, 10):
                draw.line([(wall_x_pix, y_center + wall_y_offset),
                          (wall_x_pix + 10, y_center + wall_y_offset - 3)], fill='black', width=2)
            draw.text((wall_x_pix + 15, y_center - 8), 'Wall', fill='black', font=None)
            
            # Mass positions
            mass_pixels = [x_to_pixel(self.positions[frame_idx, i]) for i in range(self.n)]
            
            # Springs
            for i in range(self.n - 1):
                draw.line([(mass_pixels[i], y_center), (mass_pixels[i+1], y_center)],
                         fill='black', width=2)
            draw.line([(mass_pixels[-1], y_center), (wall_x_pix, y_center)],
                     fill='black', width=2)
            
            # Masses
            for i in range(self.n):
                x_pix = mass_pixels[i]
                draw.ellipse([(x_pix - mass_radius, y_center - mass_radius),
                             (x_pix + mass_radius, y_center + mass_radius)],
                            fill='black', outline='black', width=2)
                label_y = y_center + mass_radius + 12
                draw.text((x_pix - 5, label_y), f'M{i+1}', fill='black', font=None)
            
            # Info
            sim_time = self.t[frame_idx]
            info_text = f'Time: {sim_time:.2f}s | Force: {self.F_input[frame_idx]:.2f}N | {self.n} masses'
            draw.text((10, 10), info_text, fill='black', font=None)
            
            # Save
            frame_path = os.path.join(temp_dir, f'frame_{frame_num:04d}.png')
            img.save(frame_path, 'PNG', optimize=False)
            
            if (frame_num + 1) % max(1, num_frames // 10) == 0:
                print(f"  {frame_num + 1}/{num_frames} frames ({100*(frame_num+1)/num_frames:.0f}%)")
        
        elapsed = time_module.time() - start_time
        print(f"✓ Frames generated in {elapsed:.1f}s")
        
        # Encode video
        print(f"Encoding video...")
        if output_file is None:
            output_file = os.path.join(temp_dir, 'animation.mp4')
        
        frame_files = sorted(Path(temp_dir).glob('frame_*.png'))
        writer = iio.get_writer(output_file, fps=fps, codec='libx264', pixelformat='yuv420p')
        
        for frame_file in frame_files:
            frame = iio.imread(str(frame_file))
            writer.append_data(frame)
        
        writer.close()
        
        video_size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"✓ Video saved: {output_file}")
        print(f"  Size: {video_size_mb:.2f} MB")
        print(f"  Duration: {num_frames/fps:.1f} seconds\n")
        
        return output_file
