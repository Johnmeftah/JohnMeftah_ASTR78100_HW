#!/usr/bin/env python3
"""
Black Hole Merger Video Creator - Production Quality
Creates smooth MP4 or GIF animation from simulation CSV data.

Features:
  - Progressive trail buildup (no pre-drawn paths)
  - 60 FPS support for butter-smooth animation
  - Real-time gravitational waveform visualization
  - Black holes scaled to actual event horizon sizes
  - Reads metadata from CSV header

Usage:
    python video_maker.py trajectory.csv [output.mp4] [--fps 60] [--duration 30]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import argparse
import subprocess
import tempfile
from scipy.io import wavfile
from scipy.interpolate import interp1d


def load_trajectory(csv_file):
    """Loads trajectory data from CSV including metadata and waveforms."""
    print(f"Loading {csv_file}...")
    
    t = []
    bh1_pos = []
    bh2_pos = []
    sep = []
    f_gw = []
    h_plus = []
    h_cross = []
    
    # metadata defaults
    metadata = {
        'm1_solar': 30.0,
        'm2_solar': 30.0,
        'r_horizon_1_m': 88600,
        'r_horizon_2_m': 88600,
        'distance_mpc': 410.0,
        'pn_order': 2.5,
        'eccentricity': 0.0
    }
    
    with open(csv_file, 'r') as f:
        for line in f:
            # parse metadata comments
            if line.startswith('#'):
                if '=' in line:
                    key, value = line[2:].strip().split('=')
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
                continue
            
            # skip empty lines
            if not line.strip():
                continue
                
            # skip header
            if line.startswith('t_sec'):
                continue
            
            parts = line.strip().split(',')
            if len(parts) >= 10:
                t.append(float(parts[0]))
                bh1_pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
                bh2_pos.append([float(parts[4]), float(parts[5]), float(parts[6])])
                sep.append(float(parts[7]))
                f_gw.append(float(parts[9]))
                # check if waveform data exists
                if len(parts) >= 12:
                    h_plus.append(float(parts[11]))
                    h_cross.append(float(parts[12]))
    
    t = np.array(t)
    bh1_pos = np.array(bh1_pos)
    bh2_pos = np.array(bh2_pos)
    sep = np.array(sep)
    f_gw = np.array(f_gw)
    h_plus = np.array(h_plus) if h_plus else None
    h_cross = np.array(h_cross) if h_cross else None
    
    print(f"  ✓ Loaded {len(t)} data points")
    print(f"  Duration: {t[-1]*1000:.1f} ms")
    print(f"  Initial separation: {sep[0]/1000:.1f} km")
    print(f"  Final separation: {sep[-1]/1000:.1f} km")
    print(f"  BH1: {metadata['m1_solar']:.1f} M☉ (horizon: {metadata['r_horizon_1_m']/1000:.1f} km)")
    print(f"  BH2: {metadata['m2_solar']:.1f} M☉ (horizon: {metadata['r_horizon_2_m']/1000:.1f} km)")
    if metadata['eccentricity'] > 0:
        print(f"  Eccentricity: {metadata['eccentricity']:.3f}")
    if h_plus is not None:
        print(f"  ✓ Waveform data included")
    
    return t, bh1_pos, bh2_pos, sep, f_gw, h_plus, h_cross, metadata


def create_video(csv_file, output_file="bh_merger.mp4", fps=60, duration=15,
                 show_waveform=True):
    """
    Creates a production-quality animated video of the black hole merger.
    
    Args:
        csv_file: input trajectory CSV
        output_file: output video file (.mp4 or .gif)
        fps: frames per second (default: 60)
        duration: video duration in seconds
        show_waveform: show gravitational waveform panel
    """
    
    # load data
    t, bh1_pos, bh2_pos, sep, f_gw, h_plus, h_cross, metadata = load_trajectory(csv_file)
    
    # check if waveform display is possible
    has_waveform = h_plus is not None and show_waveform
    
    # convert to km for better visualization
    scale = 1e-3  # meters to km
    bh1_pos_km = bh1_pos * scale
    bh2_pos_km = bh2_pos * scale
    sep_km = sep * scale
    
    # get event horizon radii in km
    r_horizon_1_km = metadata['r_horizon_1_m'] / 1000
    r_horizon_2_km = metadata['r_horizon_2_m'] / 1000
    
    # determine frame indices
    total_frames = int(fps * duration)
    frame_indices = np.linspace(0, len(t)-1, total_frames).astype(int)
    
    print(f"\n{'='*60}")
    print(f"  VIDEO CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Output:        {output_file}")
    print(f"  Duration:      {duration} sec")
    print(f"  FPS:           {fps}")
    print(f"  Total frames:  {total_frames}")
    print(f"  BH1 radius:    {r_horizon_1_km:.1f} km")
    print(f"  BH2 radius:    {r_horizon_2_km:.1f} km")
    print(f"  Waveform:      {'Yes' if has_waveform else 'No'}")
    print(f"{'='*60}\n")
    
    # set up the figure
    if has_waveform:
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                             wspace=0.15, hspace=0.25)
        ax1 = fig.add_subplot(gs[:, 0])  # orbit (left, full height)
        ax2 = fig.add_subplot(gs[0, 1])  # separation (top right)
        ax3 = fig.add_subplot(gs[1, 1])  # waveform (bottom right)
    else:
        fig = plt.figure(figsize=(14, 10))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = None
    
    fig.patch.set_facecolor('black')
    
    # style orbit plot
    ax1.set_aspect('equal')
    ax1.set_facecolor('black')
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_xlabel('x [km]', color='white', fontsize=11)
    ax1.set_ylabel('y [km]', color='white', fontsize=11)
    ax1.tick_params(colors='white', labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color('#333333')
    
    # style separation plot
    ax2.set_facecolor('black')
    ax2.grid(True, alpha=0.15, color='white')
    ax2.set_xlabel('Time [ms]', color='white', fontsize=10)
    ax2.set_ylabel('Separation [km]', color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#333333')
    # plot full separation curve (faded)
    ax2.plot(t * 1000, sep_km, color='#1a4a5e', alpha=0.4, linewidth=1)
    
    # style waveform plot
    if has_waveform:
        ax3.set_facecolor('black')
        ax3.grid(True, alpha=0.15, color='white')
        ax3.set_xlabel('Time [ms]', color='white', fontsize=10)
        ax3.set_ylabel('Strain h₊ × 10²¹', color='white', fontsize=10)
        ax3.tick_params(colors='white', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_color('#333333')
        
        # create color-coded waveform by frequency (static background)
        # use LineCollection for color gradient
        points = np.array([t * 1000, h_plus * 1e21]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # normalize frequency for colormap
        norm = plt.Normalize(f_gw.min(), f_gw.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm, alpha=0.6, linewidth=1.5)
        lc.set_array(f_gw[:-1])
        ax3.add_collection(lc)
        
        # add colorbar - more vivid
        cbar = fig.colorbar(lc, ax=ax3, pad=0.02, aspect=25)
        cbar.set_label('GW Frequency [Hz]', color='white', fontsize=10, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=9)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=9)
        
        h_max = np.max(np.abs(h_plus)) * 1e21 * 1.2
        ax3.set_xlim(0, t[-1] * 1000)
        ax3.set_ylim(-h_max, h_max)
    
    # initialize plot elements
    # trails (will be built progressively)
    trail1, = ax1.plot([], [], 'cyan', alpha=0.7, linewidth=1.2)
    trail2, = ax1.plot([], [], 'magenta', alpha=0.7, linewidth=1.2)
    
    # fading trail effect (recent positions brighter)
    trail1_fade, = ax1.plot([], [], 'cyan', alpha=0.3, linewidth=0.6)
    trail2_fade, = ax1.plot([], [], 'magenta', alpha=0.3, linewidth=0.6)
    
    # black holes
    bh1_circle = Circle((0, 0), r_horizon_1_km, color='cyan', zorder=10)
    bh2_circle = Circle((0, 0), r_horizon_2_km, color='magenta', zorder=10)
    ax1.add_patch(bh1_circle)
    ax1.add_patch(bh2_circle)
    
    # separation progress
    sep_line, = ax2.plot([], [], 'cyan', linewidth=1.5)
    sep_marker, = ax2.plot([], [], 'wo', markersize=6, zorder=10)
    
    # waveform progress (will be color-coded)
    if has_waveform:
        wave_line, = ax3.plot([], [], 'cyan', linewidth=1.2)
        wave_marker, = ax3.plot([], [], 'wo', markersize=5, zorder=10)
    
    # title and info
    title = fig.suptitle('', color='white', fontsize=13, y=0.98)
    
    # set axis limits
    max_extent = max(
        np.max(np.abs(bh1_pos_km[:, :2])),
        np.max(np.abs(bh2_pos_km[:, :2]))
    ) * 1.15
    ax1.set_xlim(-max_extent, max_extent)
    ax1.set_ylim(-max_extent, max_extent)
    
    ax2.set_xlim(0, t[-1] * 1000)
    ax2.set_ylim(0, sep_km[0] * 1.1)
    
    # trail history length (for the brighter recent trail)
    recent_trail_len = max(50, len(t) // 100)
    
    def init():
        """Initializes the animation."""
        trail1.set_data([], [])
        trail2.set_data([], [])
        trail1_fade.set_data([], [])
        trail2_fade.set_data([], [])
        bh1_circle.center = (bh1_pos_km[0, 0], bh1_pos_km[0, 1])
        bh2_circle.center = (bh2_pos_km[0, 0], bh2_pos_km[0, 1])
        sep_line.set_data([], [])
        sep_marker.set_data([], [])
        if has_waveform:
            wave_line.set_data([], [])
            wave_marker.set_data([], [])
            return trail1, trail2, trail1_fade, trail2_fade, bh1_circle, bh2_circle, sep_line, sep_marker, wave_line, wave_marker, title
        return trail1, trail2, trail1_fade, trail2_fade, bh1_circle, bh2_circle, sep_line, sep_marker, title
    
    def animate(frame):
        """Updates the animation frame."""
        idx = frame_indices[frame]
        
        # current positions
        x1, y1 = bh1_pos_km[idx, 0], bh1_pos_km[idx, 1]
        x2, y2 = bh2_pos_km[idx, 0], bh2_pos_km[idx, 1]
        
        # update black hole positions
        bh1_circle.center = (x1, y1)
        bh2_circle.center = (x2, y2)
        
        # progressive trail buildup (full trail from start to now)
        # older part (faded)
        fade_end = max(0, idx - recent_trail_len)
        if fade_end > 0:
            trail1_fade.set_data(bh1_pos_km[:fade_end, 0], bh1_pos_km[:fade_end, 1])
            trail2_fade.set_data(bh2_pos_km[:fade_end, 0], bh2_pos_km[:fade_end, 1])
        
        # recent part (brighter)
        recent_start = max(0, idx - recent_trail_len)
        trail1.set_data(bh1_pos_km[recent_start:idx+1, 0], bh1_pos_km[recent_start:idx+1, 1])
        trail2.set_data(bh2_pos_km[recent_start:idx+1, 0], bh2_pos_km[recent_start:idx+1, 1])
        
        # update separation plot
        sep_line.set_data(t[:idx+1] * 1000, sep_km[:idx+1])
        sep_marker.set_data([t[idx] * 1000], [sep_km[idx]])
        
        # update waveform plot
        if has_waveform:
            wave_line.set_data(t[:idx+1] * 1000, h_plus[:idx+1] * 1e21)
            wave_marker.set_data([t[idx] * 1000], [h_plus[idx] * 1e21])
        
        # update title
        progress = (idx / len(t)) * 100
        current_sep = sep_km[idx]
        current_freq = f_gw[idx]
        title_text = (f"Binary Black Hole Inspiral: {metadata['m1_solar']:.0f} + {metadata['m2_solar']:.0f} M☉  |  "
                     f"t = {t[idx]*1000:.1f} ms  |  r = {current_sep:.0f} km  |  f = {current_freq:.0f} Hz")
        title.set_text(title_text)
        
        # print progress
        if frame % (fps * 2) == 0:  # every 2 seconds of video
            print(f"  Rendering: {frame+1}/{total_frames} frames ({progress:.1f}%)")
        
        if has_waveform:
            return trail1, trail2, trail1_fade, trail2_fade, bh1_circle, bh2_circle, sep_line, sep_marker, wave_line, wave_marker, title
        return trail1, trail2, trail1_fade, trail2_fade, bh1_circle, bh2_circle, sep_line, sep_marker, title
    
    # create animation
    print("  Starting render...")
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=total_frames, interval=1000/fps,
                        blit=True, repeat=False)
    
    # generate audio from waveform if available
    audio_file = None
    if has_waveform:
        print("  Generating audio from waveform...")
        audio_file = generate_audio(t, h_plus, f_gw, duration, sample_rate=44100)
    
    # save animation
    print(f"  Encoding video...")
    
    if output_file.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
    else:
        # save video to temp file first
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        try:
            writer = FFMpegWriter(fps=fps, bitrate=5000, 
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            anim.save(temp_video, writer=writer)
            
            # combine video + audio if we have audio
            if audio_file and os.path.exists(audio_file):
                print("  Adding audio track...")
                try:
                    # use ffmpeg to combine
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', audio_file,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest',
                        output_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"  Warning: Could not add audio ({result.stderr[:100]})")
                        os.rename(temp_video, output_file)
                    else:
                        os.remove(temp_video)
                except Exception as e:
                    print(f"  Warning: Audio embedding failed ({e})")
                    os.rename(temp_video, output_file)
                
                # clean up audio file
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            else:
                os.rename(temp_video, output_file)
                
        except Exception as e:
            print(f"  FFmpeg not available ({e})")
            print(f"  Falling back to GIF...")
            output_file = output_file.replace('.mp4', '.gif')
            writer = PillowWriter(fps=min(fps, 30))  # gif limited to 30fps
            anim.save(output_file, writer=writer)
    
    print(f"\n{'='*60}")
    print(f"  ✓ VIDEO COMPLETE: {output_file}")
    if has_waveform and not output_file.endswith('.gif'):
        print(f"  ✓ Audio embedded (turn up your volume!)")
    print(f"{'='*60}\n")
    
    plt.close()
    return output_file


def generate_audio(t, h_plus, f_gw, duration, sample_rate=44100, freq_shift=20.0):
    """
    Generates audio from gravitational waveform with frequency shifting.
    
    Shifts frequencies UP so early orbits (~3 Hz) become audible (~60+ Hz).
    Creates the "wooo... wooo... WOOOP!" chirp sound.
    
    Args:
        t, h_plus, f_gw: waveform data
        duration: output duration in seconds
        sample_rate: audio sample rate
        freq_shift: frequency multiplier (20 = 3Hz becomes 60Hz)
    
    Returns:
        path to temporary WAV file
    """
    # original signal properties
    t_original = t[-1] - t[0]
    
    # time stretch factor
    time_stretch = duration / t_original
    
    # for proper frequency shifting, we need to:
    # 1. Compute the phase of the waveform
    # 2. Multiply the phase by freq_shift
    # 3. Reconstruct the waveform with shifted phase
    
    # get amplitude envelope
    from scipy.signal import hilbert
    analytic = hilbert(h_plus)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    
    # create output time array
    n_output = int(duration * sample_rate)
    t_out = np.linspace(0, duration, n_output)
    
    # map to input time
    t_input_mapped = t_out / time_stretch
    
    # normalize times
    t_norm = (t - t[0]) / t_original
    t_input_norm = np.clip(t_input_mapped / t_original, 0, 1)
    
    # interpolate amplitude and phase
    amp_interp = interp1d(t_norm, amplitude, kind='linear', fill_value='extrapolate')
    phase_interp = interp1d(t_norm, phase, kind='linear', fill_value='extrapolate')
    
    amp_out = amp_interp(t_input_norm)
    phase_out = phase_interp(t_input_norm)
    
    # apply frequency shift by multiplying phase
    phase_shifted = phase_out * freq_shift
    
    # reconstruct waveform
    h_shifted = amp_out * np.cos(phase_shifted)
    
    # normalize to [-1, 1]
    h_max = np.max(np.abs(h_shifted))
    if h_max > 0:
        h_audio = h_shifted / h_max * 0.85
    else:
        h_audio = h_shifted
    
    # apply fade in/out
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    if fade_samples > 0 and len(h_audio) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        h_audio[:fade_samples] *= fade_in
        h_audio[-fade_samples:] *= fade_out
    
    # convert to 16-bit PCM
    h_16bit = (h_audio * 32767).astype(np.int16)
    
    # save to temp file
    audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    wavfile.write(audio_file, sample_rate, h_16bit)
    
    print(f"    Original freq: {f_gw[0]:.0f} - {f_gw[-1]:.0f} Hz")
    print(f"    Shifted freq:  {f_gw[0]*freq_shift:.0f} - {f_gw[-1]*freq_shift:.0f} Hz")
    
    return audio_file


def main():
    parser = argparse.ArgumentParser(
        description='Create production-quality black hole merger animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # standard quality (30 fps, 10 sec)
  python video_maker.py inspiral.csv

  # high quality (60 fps, 20 sec, smooth!)
  python video_maker.py inspiral.csv --fps 60 --duration 20

  # ultra smooth (60 fps, 30 sec)
  python video_maker.py inspiral.csv merger.mp4 -f 60 -d 30

  # gif output (limited to 30 fps)
  python video_maker.py inspiral.csv output.gif -d 10
        '''
    )
    
    parser.add_argument('csv_file', help='Input trajectory CSV file')
    parser.add_argument('output', nargs='?', default='bh_merger.mp4',
                       help='Output video file (default: bh_merger.mp4)')
    parser.add_argument('--duration', '-d', type=float, default=15,
                       help='Video duration in seconds (default: 15)')
    parser.add_argument('--fps', '-f', type=int, default=60,
                       help='Frames per second (default: 60)')
    parser.add_argument('--no-waveform', action='store_true',
                       help='Hide gravitational waveform panel')
    
    args = parser.parse_args()
    
    # check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found!")
        print("Make sure you've run the simulation first.")
        sys.exit(1)
    
    # create video
    try:
        create_video(
            args.csv_file, 
            args.output, 
            fps=args.fps, 
            duration=args.duration,
            show_waveform=not args.no_waveform
        )
    except Exception as e:
        print(f"\nError creating video: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Make sure matplotlib is installed: pip install matplotlib")
        print("  - For MP4, you need ffmpeg: brew install ffmpeg (Mac) or apt install ffmpeg (Linux)")
        print("  - Or use .gif format: python video_maker.py data.csv output.gif")
        sys.exit(1)


if __name__ == "__main__":
    main()
