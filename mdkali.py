import numpy as np
import cv2
import moviepy
from moviepy.video.VideoClip import VideoClip
import moviepy.video.io.ImageSequenceClip
from moviepy.video.VideoClip import ImageClip
import moviepy.video.fx.all as vfx
from moviepy.audio.io.AudioFileClip import AudioFileClip
import os
from scipy.fftpack import fft
from scipy.signal import spectrogram
from scipy.io import wavfile
import argparse
from typing import Tuple
import logging
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm
import glob
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import base64
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_album_art(mp3_path: str) -> bool:
    """Extract album art from MP3 file if present."""
    try:
        audio = MP3(mp3_path, ID3=ID3)
        if audio.tags:
            for tag in audio.tags.values():
                if tag.FrameID in ['APIC', 'PIC']:
                    img_data = tag.data
                    img = Image.open(io.BytesIO(img_data))
                    output_path = os.path.splitext(mp3_path)[0] + '.' + img.format.lower()
                    img.save(output_path)
                    return True
        return False
    except Exception as e:
        logger.error(f"Error extracting album art: {str(e)}")
        return False
    
class AudioVisualizer:
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Width, height, and fps must be positive numbers")
            
        self.width = width
        self.height = height
        self.fps = fps
        self.visualization_functions = {
            'kaleidoscope': self.kaleidoscope_visualization,
            'waveform': self.waveform_visualization,
            'spectrum': self.spectrum_visualization,
            'circular_spectrum': self.circular_spectrum_visualization,
            'dancing_particles': self.dancing_particles_visualization,
            'bars_3d': self.bars_3d_visualization,
            'pulse': self.pulse_visualization,
            'equalizer': self.equalizer_visualization,
            'spectrogram': self.spectrogram_visualization,
            'MD_spectrogram': self.MD_spectrogram_visualization
        }
    
        self.center = (self.width // 2, self.height // 2)
        self.max_radius = min(self.width, self.height) // 4

        # Initialize particle system for dancing particles
        self.particles = []
        self.init_particles(100)  # Initialize 100 particles
        
        # Initialize spectrogram buffer
        self.spec_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def init_particles(self, num_particles: int):
        """Initialize particles for dancing particles visualization."""
        self.particles = []
        for _ in range(num_particles):
            self.particles.append({
                'x': np.random.randint(0, self.width),
                'y': np.random.randint(0, self.height),
                'vx': np.random.randn() * 2,
                'vy': np.random.randn() * 2,
                'size': np.random.randint(3, 10)
            })

    def reset_buffers(self):
        """Reset all visualization buffers between songs."""
        self.spec_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if hasattr(self, 'md_spec_buffers'):
            quarter_width = (self.width - 3 * int(min(self.width, self.height) * 0.04)) // 2
            quarter_height = (self.height - 3 * int(min(self.width, self.height) * 0.04)) // 2
            self.md_spec_buffers = {
                'left': np.zeros((quarter_height, quarter_width, 3), dtype=np.uint8),
                'right': np.zeros((quarter_height, quarter_width, 3), dtype=np.uint8)
            }    

    def convert_to_wav(self, mp3_path: str) -> str:
        """Convert MP3 to WAV using ffmpeg."""
        wav_path = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run([
                'ffmpeg', '-i', mp3_path,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1',
                wav_path
            ], check=True, capture_output=True)
            return wav_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting MP3 to WAV: {e.stderr.decode()}")
            raise

    def get_audio_data(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio data from audio file. Modified to preserve stereo channels."""
        try:
            # Convert MP3 to WAV if necessary
            if audio_path.lower().endswith('.mp3'):
                wav_path = self.convert_to_wav(audio_path)
            else:
                wav_path = audio_path

            # Read WAV file
            sample_rate, samples = wavfile.read(wav_path)
            
            # Convert to float32 and normalize, preserving channels
            samples = samples.astype(np.float32)
            max_val = np.max(np.abs(samples))
            if max_val > 0:  # Avoid division by zero
                samples = samples / max_val

            # Clean up temporary file
            if wav_path != audio_path:
                try:
                    os.remove(wav_path)
                except:
                    pass

            return samples, sample_rate
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise

    def create_frame(self, t: float, audio_samples: np.ndarray, sample_rate: int, 
                    visualization_type: str) -> np.ndarray:
        """Create a frame at time t."""
        if visualization_type not in self.visualization_functions:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        try:
            return self.visualization_functions[visualization_type](t, audio_samples, sample_rate)
        except Exception as e:
            logger.error(f"Error creating frame at time {t}: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)            

    def kaleidoscope_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create kaleidoscope effect based on audio amplitude."""
        try:
            # Calculate the current audio segment
            start_idx = int(t * sample_rate)
            end_idx = min(start_idx + sample_rate // self.fps, len(audio_samples))
            if start_idx >= len(audio_samples):
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            current_samples = audio_samples[start_idx:end_idx]
            amplitude = np.clip(np.abs(current_samples).mean(), 0, 1)
            
            # Create base image
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Scale amplitude to reasonable values and clip to max radius
            radius = int(self.max_radius * amplitude) + 50
            radius = min(radius, self.max_radius)
            
            num_points = 8
            period = 5.0  # Controls animation speed
            
            # Create kaleidoscope pattern
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = int(self.center[0] + radius * np.cos(angle + t / period))
                y = int(self.center[1] + radius * np.sin(angle + t / period))
                
                # Ensure points are within frame bounds
                x = np.clip(x, 0, self.width - 1)
                y = np.clip(y, 0, self.height - 1)
                
                color = np.array([
                    int(127 + 127 * np.sin(t / period + i)),
                    int(127 + 127 * np.sin(t / period + i + 2*np.pi/3)),
                    int(127 + 127 * np.sin(t / period + i + 4*np.pi/3))
                ], dtype=np.uint8)
                
                cv2.circle(frame, (x, y), radius, color.tolist(), -1)
            
            # Apply blur for smooth effect
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
            return frame
            
        except Exception as e:
            logger.error(f"Error in kaleidoscope visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def waveform_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create waveform visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Calculate the current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            samples = audio_samples[start_idx:end_idx]
            
            # Ensure we have enough samples to work with
            if len(samples) < 2:
                return frame
            
            # Scale samples to fit the frame height with padding
            padding = 0.1  # 10% padding top and bottom
            scale_factor = (self.height * (1 - 2*padding)) / 2
            scaled_samples = samples * scale_factor
            
            # Draw waveform
            points = []
            x_scale = self.width / (len(samples) - 1)
            for i, sample in enumerate(scaled_samples):
                x = int(i * x_scale)
                y = int(self.height/2 + sample)
                y = np.clip(y, 0, self.height - 1)
                points.append((x, y))
            
            # Draw lines with anti-aliasing
            for i in range(len(points)-1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2, cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in waveform visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def spectrum_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create frequency spectrum visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Calculate the current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            # Apply Hanning window to reduce spectral leakage
            samples = audio_samples[start_idx:end_idx] * np.hanning(end_idx - start_idx)
            
            # Compute FFT and get magnitude spectrum
            spectrum = np.abs(fft(samples))
            
            # Use only first half of the spectrum (due to symmetry)
            spectrum = spectrum[:len(spectrum)//2]
            
            # Apply log scaling to better represent audio spectrum
            spectrum = np.log10(spectrum + 1)
            
            # Normalize spectrum
            max_magnitude = max(1e-10, np.max(spectrum))
            scaled_spectrum = spectrum * (self.height*0.8) / max_magnitude
            
            # Draw spectrum bars
            num_bars = min(len(scaled_spectrum), self.width)
            bar_width = max(1, self.width // num_bars)
            
            for i in range(num_bars):
                x = i * bar_width
                y = int(scaled_spectrum[i])
                y = np.clip(y, 0, self.height)
                
                # Color based on frequency (blue->green->red)
                hue = int(180 * (1 - i / num_bars))
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                
                cv2.rectangle(frame, 
                             (x, self.height), 
                             (x + bar_width - 1, self.height - y),
                             color.tolist(),
                             -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in spectrum visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def circular_spectrum_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create circular spectrum analyzer visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Get current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            # Compute spectrum
            samples = audio_samples[start_idx:end_idx] * np.hanning(end_idx - start_idx)
            spectrum = np.abs(fft(samples))[:len(samples)//2]
            spectrum = np.log10(spectrum + 1)
            
            # Normalize spectrum
            spectrum = spectrum / max(1e-10, np.max(spectrum))
            
            # Draw circular bars
            num_bars = 180
            bar_width = 2
            for i in range(num_bars):
                angle = i * 2 * np.pi / num_bars
                idx = int(i * len(spectrum) / num_bars)
                height = int(self.max_radius * spectrum[idx])
                
                start_point = (
                    int(self.center[0] + (self.max_radius - height) * np.cos(angle)),
                    int(self.center[1] + (self.max_radius - height) * np.sin(angle))
                )
                end_point = (
                    int(self.center[0] + self.max_radius * np.cos(angle)),
                    int(self.center[1] + self.max_radius * np.sin(angle))
                )
                
                # Color based on frequency
                hue = int(180 * (1 - i / num_bars))
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                
                cv2.line(frame, start_point, end_point, color.tolist(), bar_width)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in circular spectrum visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def dancing_particles_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create dancing particles visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Get current audio segment and its energy
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            samples = audio_samples[start_idx:end_idx]
            energy = np.sqrt(np.mean(samples**2))
            
            # Update and draw particles
            for particle in self.particles:
                # Update position based on velocity
                particle['x'] += particle['vx'] * (1 + energy * 10)
                particle['y'] += particle['vy'] * (1 + energy * 10)
                
                # Bounce off edges
                if particle['x'] < 0 or particle['x'] >= self.width:
                    particle['vx'] *= -1
                if particle['y'] < 0 or particle['y'] >= self.height:
                    particle['vy'] *= -1
                
                # Keep particles in bounds
                particle['x'] = np.clip(particle['x'], 0, self.width - 1)
                particle['y'] = np.clip(particle['y'], 0, self.height - 1)
                
                # Draw particle with size based on energy
                size = int(particle['size'] * (1 + energy * 5))
                color = (
                    int(255 * energy),
                    int(149 * energy),
                    int(43 * energy)
                )
                cv2.circle(frame, 
                          (int(particle['x']), int(particle['y'])), 
                          size, 
                          color, 
                          -1)
                
                # Draw connections between nearby particles
                for other in self.particles:
                    dx = particle['x'] - other['x']
                    dy = particle['y'] - other['y']
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 100:  # Connection threshold
                        alpha = 1 - dist/100
                        color = (
                            int(255 * energy * alpha),
                            int(149 * energy * alpha),
                            int(43 * energy * alpha)
                        )
                        cv2.line(frame,
                                (int(particle['x']), int(particle['y'])),
                                (int(other['x']), int(other['y'])),
                                color,
                                1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in dancing particles visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def bars_3d_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create 3D bars visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Get current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            # Compute spectrum
            samples = audio_samples[start_idx:end_idx] * np.hanning(end_idx - start_idx)
            spectrum = np.abs(fft(samples))[:len(samples)//2]
            spectrum = np.log10(spectrum + 1)
            
            # Normalize spectrum
            spectrum = spectrum / max(1e-10, np.max(spectrum))
            
            # Parameters for 3D perspective
            num_bars = 16
            bar_width = self.width // (num_bars * 2)
            perspective_factor = 0.3
            
            # Draw bars with 3D effect
            for i in range(num_bars):
                idx = int(i * len(spectrum) / num_bars)
                height = int(self.height * 0.8 * spectrum[idx])
                
                # Calculate 3D coordinates
                x_base = self.width//4 + i * bar_width * 2
                x_top = x_base + int(height * perspective_factor)
                
                # Draw 3D bar
                points = np.array([
                    [x_base, self.height],  # bottom left
                    [x_base + bar_width, self.height],  # bottom right
                    [x_top + bar_width, self.height - height],  # top right
                    [x_top, self.height - height]  # top left
                ], np.int32)
                
                # Color based on height
                color = (
                    int(255 * spectrum[idx]),
                    int(149 * spectrum[idx]),
                    int(43 * spectrum[idx])
                )
                
                # Draw filled polygon
                cv2.fillPoly(frame, [points], color)
                
                # Draw edges for 3D effect
                cv2.polylines(frame, [points], True, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in 3D bars visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def pulse_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create pulsing glow visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Get current audio segment and energy
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            samples = audio_samples[start_idx:end_idx]
            energy = np.sqrt(np.mean(samples**2))
            
            # Create multiple pulses with different sizes
            num_pulses = 5
            base_size = min(self.width, self.height) // 4
            
            for i in range(num_pulses):
                size = int(base_size * (1 + energy * 5) * (1 + i/2))
                alpha = 1 - (i / num_pulses)
                color = (
                    int(255 * energy * alpha),
                    int(149 * energy * alpha),
                    int(43 * energy * alpha)
                )
                
                cv2.circle(frame,
                          self.center,
                          size,
                          color,
                          -1)
                
                # Add glow effect
                kernel_size = 21
                frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in pulse visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def equalizer_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create music equalizer visualization."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Get current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            # Compute spectrum
            samples = audio_samples[start_idx:end_idx] * np.hanning(end_idx - start_idx)
            spectrum = np.abs(fft(samples))[:len(samples)//2]
            spectrum = np.log10(spectrum + 1)
            
            # Normalize spectrum
            spectrum = spectrum / max(1e-10, np.max(spectrum))
            
            # Parameters for equalizer
            num_bands = 32
            bar_width = (self.width - (num_bands + 1) * 10) // num_bands
            bar_spacing = 10
            
            # Calculate band averages
            bands = np.array_split(spectrum, num_bands)
            band_energies = [np.mean(band) for band in bands]
            
            # Draw equalizer bars
            for i, energy in enumerate(band_energies):
                x = i * (bar_width + bar_spacing) + bar_spacing
                height = int(self.height * 0.8 * energy)
                
                # Main bar
                color = (
                    int(255 * energy),
                    int(149 * energy),
                    int(43 * energy)
                )
                
                cv2.rectangle(frame,
                            (x, self.height - height),
                            (x + bar_width, self.height),
                            color,
                            -1)
                
                # Peak marker
                peak_height = 5
                cv2.rectangle(frame,
                            (x, self.height - height - peak_height),
                            (x + bar_width, self.height - height),
                            (255, 255, 255),
                            -1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in equalizer visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def spectrogram_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create scrolling spectrogram visualization with corrected colors and max frequency of 12kHz."""
        try:
            # Get current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return self.spec_buffer

            # Compute spectrum for current window
            samples = audio_samples[start_idx:end_idx] * np.hanning(end_idx - start_idx)
            spectrum = np.abs(fft(samples))[:len(samples)//2]
            spectrum = np.log10(spectrum + 1)
            
            # Calculate the index corresponding to 12kHz
            max_freq_idx = int(12000 * len(spectrum) / (sample_rate/2))
            spectrum = spectrum[:max_freq_idx]
            
            # Normalize spectrum
            spectrum = spectrum / max(1e-10, np.max(spectrum))
            
            # Resize spectrum to match height
            num_freqs = self.height
            spectrum_resized = cv2.resize(spectrum, (1, num_freqs))
            
            # Shift existing buffer left
            self.spec_buffer = np.roll(self.spec_buffer, -1, axis=1)
            
            # Convert spectrum to color values using the specified color palette
            for i, magnitude in enumerate(spectrum_resized):
                # Interpolate between min and max colors (BGR format)
                # min_color: RGB(32,0,0) -> BGR(0,0,32)
                # max_color: RGB(255,149,43) -> BGR(43,149,255)
                b = int(0 + magnitude * 43)           # 0 to 43
                g = int(0 + magnitude * 149)          # 0 to 149
                r = int(32 + magnitude * (255 - 32))  # 32 to 255
                
                # Update the rightmost column of the buffer
                self.spec_buffer[num_freqs-1-i, -1] = [r, g, b]
            
            return self.spec_buffer
            
        except Exception as e:
            logger.error(f"Error in spectrogram visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def MD_spectrogram_visualization(self, t: float, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Create multi-display spectrogram visualization with waveforms and spectrograms for both channels."""
        try:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Calculate border size (2% of minimum dimension)
            border = int(min(self.width, self.height) * 0.04)
            
            # Calculate quarter dimensions
            quarter_width = (self.width - 3 * border) // 2
            quarter_height = (self.height - 3 * border) // 2
            
            # Create sub-visualizers for each quarter if they don't exist
            if not hasattr(self, 'md_spec_buffers'):
                self.md_spec_buffers = {
                    'left': np.zeros((quarter_height, quarter_width, 3), dtype=np.uint8),
                    'right': np.zeros((quarter_height, quarter_width, 3), dtype=np.uint8)
                }
            
            # Get current audio segment
            start_idx = int(t * sample_rate)
            window_size = sample_rate // self.fps
            end_idx = min(start_idx + window_size, len(audio_samples))
            
            if start_idx >= len(audio_samples):
                return frame
            
            # Split stereo channels
            if len(audio_samples.shape) > 1:
                left_channel = audio_samples[start_idx:end_idx, 0]
                right_channel = audio_samples[start_idx:end_idx, 1]
            else:
                # If mono, use the same for both channels
                left_channel = right_channel = audio_samples[start_idx:end_idx]
            
            # Draw waveforms
            def draw_waveform(samples, x_offset, y_offset, width, height):
                points = []
                x_scale = width / (len(samples) - 1)
                scale_factor = height / 2 * 0.8  # 80% of half height
                
                for i, sample in enumerate(samples):
                    x = int(x_offset + i * x_scale)
                    y = int(y_offset + height/2 + sample * scale_factor)
                    y = np.clip(y, y_offset, y_offset + height)
                    points.append((x, y))
                
                for i in range(len(points)-1):
                    cv2.line(frame, points[i], points[i+1], (0, 255, 0), 1, cv2.LINE_AA)
            
            # Draw spectrograms
            def update_spectrogram(samples, buffer):
                # Compute spectrum
                spectrum = np.abs(fft(samples * np.hanning(len(samples))))[:len(samples)//2]
                spectrum = np.log10(spectrum + 1)
                
                # Limit to 12kHz
                max_freq_idx = int(12000 * len(spectrum) / (sample_rate/2))
                spectrum = spectrum[:max_freq_idx]
                
                # Normalize
                spectrum = spectrum / max(1e-10, np.max(spectrum))
                
                # Resize to match height
                spectrum_resized = cv2.resize(spectrum, (1, buffer.shape[0]))
                
                # Roll buffer left
                buffer = np.roll(buffer, -1, axis=1)
                
                # Update rightmost column with correct colors
                for i, magnitude in enumerate(spectrum_resized):
                    b = int(0 + magnitude * 43)
                    g = int(0 + magnitude * 149)
                    r = int(32 + magnitude * (255 - 32))
                    buffer[buffer.shape[0]-1-i, -1] = [r, g, b]
                
                return buffer
            
            # Draw left channel
            draw_waveform(left_channel, border, border, quarter_width, quarter_height)
            self.md_spec_buffers['left'] = update_spectrogram(
                left_channel, self.md_spec_buffers['left']
            )
            
            # Draw right channel
            draw_waveform(right_channel, 2*border + quarter_width, border, 
                         quarter_width, quarter_height)
            self.md_spec_buffers['right'] = update_spectrogram(
                right_channel, self.md_spec_buffers['right']
            )
            
            # Copy spectrograms to frame
            frame[border*2 + quarter_height:border*2 + quarter_height*2, 
                 border:border + quarter_width] = self.md_spec_buffers['left']
            frame[border*2 + quarter_height:border*2 + quarter_height*2, 
                 border*2 + quarter_width:border*2 + quarter_width*2] = self.md_spec_buffers['right']
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in MD spectrogram visualization: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def create_video(self, audio_path: str, output_path: str, visualization_type: str = 'kaleidoscope'):
        """Create video with audio visualization."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            logger.info("Processing audio file...")
            audio_samples, sample_rate = self.get_audio_data(audio_path)
            
            duration = len(audio_samples) / sample_rate
            
            # Create a progress bar for frame generation
            total_frames = int(duration * self.fps)
            frame_count = [0]  # Use list to allow modification in closure
            progress_bar = tqdm(total=total_frames, desc="Generating frames", unit="frame")
            
            def make_frame(t):
                frame = self.create_frame(t, audio_samples, sample_rate, visualization_type)
                frame_count[0] += 1
                progress_bar.update(1)
                return frame
            
            logger.info("Creating video clip...")
            clip = VideoClip(make_frame, duration=duration)
            clip.fps = self.fps
            
            logger.info("Adding audio to video...")
            audio = AudioFileClip(audio_path)
            final_clip = clip.set_audio(audio)
            
            logger.info("Writing video file...")
            final_clip.write_videofile(
                output_path,
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=4,
                logger=None  # Disable moviepy's logger
            )
            
            # Clean up
            progress_bar.close()
            clip.close()
            audio.close()
            final_clip.close()
            
            logger.info("Video creation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise            

def main():
    parser = argparse.ArgumentParser(description='Convert MP3 to MP4 with audio visualization')
    parser.add_argument('input_files', nargs='+', help='Input MP3 file(s) (wildcards supported)')
    parser.add_argument('--output', '-o', help='Output MP4 file (ignored if multiple input files)')
    parser.add_argument('--visualization', '-v', 
                       choices=['kaleidoscope', 'waveform', 'spectrum', 
                               'circular_spectrum', 'dancing_particles', 
                               'bars_3d', 'pulse', 'equalizer', 'spectrogram',
                               'MD_spectrogram'],
                       default='MD_spectrogram',
                       help='Visualization type')
    parser.add_argument('--width', '-w', type=int, default=1920, help='Video width')
    parser.add_argument('--height', '-H', type=int, default=1080, help='Video height')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Frames per second')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--extract-art', '-e', action='store_true', help='Extract album art')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Expand wildcards
    input_files = []
    for pattern in args.input_files:
        input_files.extend(glob.glob(pattern))
    
    if not input_files:
        logger.error("No input files found")
        exit(1)
    
    visualizer = AudioVisualizer(args.width, args.height, args.fps)
    
    for input_file in input_files:
        try:
            if args.extract_art:
                extract_album_art(input_file)
            
            output_file = args.output if len(input_files) == 1 and args.output else \
                         os.path.splitext(input_file)[0] + '.mp4'
            
            visualizer.reset_buffers()  # Reset buffers before processing each file
            visualizer.create_video(input_file, output_file, args.visualization)
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()        