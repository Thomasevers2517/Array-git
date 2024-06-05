""" WORK IN PROGRESS
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_room_with_animation(sources, microphones, room_dimensions=(10, 10), num_frames=200, interval=50, wave_lifespan=20):
    """
    Plots and animates a top-view of a room with sources and microphones.
    Sound waves are animated as expanding circles from the sources and bouncing off the walls.
    
    Parameters:
    - sources: List of tuples representing the (x, y) positions of the sources.
    - microphones: List of tuples representing the (x, y) positions of the microphones.
    - room_dimensions: Tuple representing the dimensions of the room (width, height).
    - num_frames: Number of frames for the animation.
    - interval: Time between frames in milliseconds.
    - wave_lifespan: Number of frames each wave lasts before dying out.
    """
    
    fig, ax = plt.subplots()
    
    # Set room dimensions to be from 0 to 10 in both directions
    ax.set_xlim(0, room_dimensions[0])
    ax.set_ylim(0, room_dimensions[1])
    
    # Plot sources and microphones
    for source in sources:
        ax.plot(source[0], source[1], 'o', label='Source' if 'Source' not in [text.get_text() for text in ax.texts] else "")
    for mic in microphones:
        ax.plot(mic[0], mic[1], 'x', label='Microphone' if 'Microphone' not in [text.get_text() for text in ax.texts] else "")
    
    # Adding labels and legend
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Top-View of Room with Sources and Microphones')
    ax.legend()
    ax.grid(True)

    # Prepare for animation
    wave_patches = []
    wave_info = []
    for source in sources:
        wave_patch = plt.Circle(source, 0, fill=False, color='blue')
        ax.add_patch(wave_patch)
        wave_patches.append(wave_patch)
        wave_info.append({
            'center': np.array(source, dtype=float),
            'radius': 0,
            'direction': np.ones(2),
            'lifespan': wave_lifespan
        })
    
    def update(frame):
        for i, wave_patch in enumerate(wave_patches):
            info = wave_info[i]
            
            if info['lifespan'] > 0:
                # Update radius
                info['radius'] += min(room_dimensions) / wave_lifespan
                
                # Check for collisions with walls and create new wave center if needed
                if info['center'][0] + info['radius'] > room_dimensions[0] or info['center'][0] - info['radius'] < 0:
                    info['direction'][0] *= -1
                    info['center'][0] = max(min(info['center'][0], room_dimensions[0]), 0)
                    info['radius'] = 0
                if info['center'][1] + info['radius'] > room_dimensions[1] or info['center'][1] - info['radius'] < 0:
                    info['direction'][1] *= -1
                    info['center'][1] = max(min(info['center'][1], room_dimensions[1]), 0)
                    info['radius'] = 0
                
                # Update wave patch
                wave_patch.center = info['center']
                wave_patch.set_radius(info['radius'])
                
                # Decrease lifespan
                info['lifespan'] -= 1
            else:
                # Reset wave patch and info if the wave dies out
                wave_patch.set_radius(0)
                info['radius'] = 0
                info['center'] = np.array(sources[i], dtype=float)
                info['direction'] = np.ones(2)
                info['lifespan'] = wave_lifespan
        
        return wave_patches

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    plt.show()

# Example usage
sources = [(2, 3), (5, 7), (8, 2)]  # Example source positions
microphones = [(1, 1), (1, 8), (9, 1), (9, 8)]  # Example microphone positions

plot_room_with_animation(sources, microphones)
