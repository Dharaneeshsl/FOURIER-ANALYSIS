import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_image_fft(image_path, display_plots=True, peak_prominence_threshold=0.1):
    """
    Analyzes an image using Fourier Transform to look for potential AI-generated artifacts.

    Args:
        image_path (str): Path to the image file.
        display_plots (bool): Whether to display the analysis plots.
        peak_prominence_threshold (float): Relative prominence for detecting peaks in the radial profile.
                                           A higher value means only more significant peaks are detected.

    Returns:
        dict: A dictionary containing analysis results.
              - 'image_dimensions': (height, width)
              - 'has_suspicious_peaks': bool
              - 'suspicious_peak_details': list of (radius, magnitude) for detected peaks
              - 'radial_profile': 1D array of azimuthally averaged spectrum
              - 'frequencies': Corresponding frequencies for the radial profile
              - 'log_magnitude_spectrum': 2D log magnitude spectrum (for visualization)
    """
    try:
        # 1. Load Image and Convert to Grayscale
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = gray_img.shape
        
        # 2. Perform 2D Discrete Fourier Transform (DFT)
        dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # 3. Shift the zero-frequency component to the center
        dft_shifted = np.fft.fftshift(dft)
        
        # 4. Calculate Magnitude Spectrum (log scale for visualization)
        # dft_shifted has 2 channels: real and imaginary.
        # Magnitude = sqrt(real^2 + imag^2)
        magnitude_spectrum = cv2.magnitude(dft_shifted[:,:,0], dft_shifted[:,:,1])
        
        # Add 1 to avoid log(0) errors, then apply log
        log_magnitude_spectrum = np.log1p(magnitude_spectrum)

        # 5. Azimuthal Averaging (Radial Profile)
        # Create a grid of distances from the center
        crow, ccol = rows // 2 , cols // 2
        y_indices, x_indices = np.ogrid[:rows, :cols]
        
        # Create a mask of radii
        r_grid = np.sqrt((x_indices - ccol)**2 + (y_indices - crow)**2)
        r_grid_int = r_grid.astype(int)

        # Calculate the sum of magnitudes for each radius
        # np.bincount sums up weights (magnitude_spectrum) for each bin index (r_grid_int)
        # minlength ensures the output array is large enough for the max radius
        max_radius = int(r_grid.max()) + 1
        radial_sum = np.bincount(r_grid_int.ravel(), weights=magnitude_spectrum.ravel(), minlength=max_radius)
        
        # Calculate the count of pixels for each radius
        radial_count = np.bincount(r_grid_int.ravel(), minlength=max_radius)
        
        # Avoid division by zero for radii with no pixels (shouldn't happen if minlength is set correctly)
        radial_profile = np.zeros_like(radial_sum)
        non_zero_counts = radial_count > 0
        radial_profile[non_zero_counts] = radial_sum[non_zero_counts] / radial_count[non_zero_counts]

        # Frequencies corresponding to the radial profile (normalized from 0 to 0.5)
        # The maximum frequency corresponds to Nyquist frequency (half the sampling rate)
        # Here, radius is analogous to frequency.
        frequencies = np.arange(len(radial_profile)) * (0.5 / (len(radial_profile) -1 if len(radial_profile) > 1 else 1))


        # 6. Heuristic Analysis: Look for suspicious peaks in the radial profile
        # Exclude the DC component (first few points) as it's always the largest
        search_start_index = min(5, len(radial_profile) -1) # Start searching after a few initial points
        
        # Normalize profile for peak finding (excluding DC for normalization range)
        if len(radial_profile[search_start_index:]) > 0:
            profile_for_peaks = radial_profile[search_start_index:]
            max_val = profile_for_peaks.max()
            if max_val > 0:
                normalized_profile_for_peaks = profile_for_peaks / max_val
            else:
                normalized_profile_for_peaks = profile_for_peaks # all zeros

            # Prominence is a measure of how much a peak stands out from the surrounding signal.
            # Use a relative prominence based on the normalized profile.
            peaks, properties = find_peaks(normalized_profile_for_peaks, prominence=peak_prominence_threshold)
            
            # Adjust peak indices back to original radial_profile indices
            detected_peaks_indices = peaks + search_start_index
        else:
            detected_peaks_indices = np.array([])
            properties = {'prominences': []}

        has_suspicious_peaks = len(detected_peaks_indices) > 0
        suspicious_peak_details = []
        if has_suspicious_peaks:
            print(f"Found {len(detected_peaks_indices)} potential suspicious peak(s) in the frequency spectrum.")
            for i, peak_idx in enumerate(detected_peaks_indices):
                peak_radius = frequencies[peak_idx] # or peak_idx itself if you prefer pixel distance
                peak_magnitude = radial_profile[peak_idx]
                peak_prominence = properties['prominences'][i]
                suspicious_peak_details.append({
                    "radius_pixels": int(peak_idx),
                    "frequency_normalized": float(peak_radius),
                    "magnitude": float(peak_magnitude),
                    "prominence_normalized": float(peak_prominence)
                })
                print(f"  - Peak at radius ~{peak_idx} pixels (norm. freq: {peak_radius:.3f}), "
                      f"Magnitude: {peak_magnitude:.2e}, Prominence: {peak_prominence:.3f}")
        else:
            print("No distinct suspicious peaks found in the frequency spectrum with current settings.")

        # 7. Visualization (Optional)
        if display_plots:
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Show original in color
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(log_magnitude_spectrum, cmap='viridis') # Use a colormap like viridis or jet
            plt.title('Log Magnitude Spectrum')
            plt.colorbar(label='Log Magnitude')
            plt.axis('off')

            plt.subplot(133)
            plt.plot(frequencies, radial_profile)
            plt.plot(frequencies[detected_peaks_indices], radial_profile[detected_peaks_indices], "x", color='red', markersize=8, label="Suspicious Peaks")
            plt.xlabel('Normalized Spatial Frequency (Radius)')
            plt.ylabel('Average Magnitude')
            plt.title('Azimuthally Averaged Radial Profile')
            plt.yscale('log') # Often better to view spectrum on log scale
            plt.grid(True, which="both", ls="-", alpha=0.5)
            if has_suspicious_peaks:
                plt.legend()
            
            plt.tight_layout()
            plt.show()

        return {
            'image_dimensions': (rows, cols),
            'has_suspicious_peaks': has_suspicious_peaks,
            'suspicious_peak_details': suspicious_peak_details,
            'radial_profile': radial_profile,
            'frequencies': frequencies,
            'log_magnitude_spectrum': log_magnitude_spectrum
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- How to Use ---
if __name__ == "__main__":
    # --- Create dummy images for testing ---
    # You should replace these with actual paths to your images.
    # For demonstration, let's create a simple "real-like" and an "AI-like" image.

    def create_test_image(filename, size=(256, 256), add_grid=False, noise_level=20):
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 128 # Gray background
        
        # Add some random noise (more characteristic of real images)
        noise = np.random.randint(-noise_level, noise_level + 1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some larger random shapes
        for _ in range(5):
            x1, y1 = np.random.randint(0, size[1]-20), np.random.randint(0, size[0]-20)
            x2, y2 = x1 + np.random.randint(10,20), y1 + np.random.randint(10,20)
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
            
        if add_grid: # Simulate a subtle periodic artifact
            for i in range(0, size[0], 16): # Every 16 pixels
                cv2.line(img, (0, i), (size[1], i), (120, 120, 120), 1) # Faint horizontal
                cv2.line(img, (i, 0), (i, size[0]), (120, 120, 120), 1) # Faint vertical
        
        cv2.imwrite(filename, img)
        print(f"Created test image: {filename}")

    create_test_image("./realimganalysis.png", add_grid=False, noise_level=30)
    create_test_image("./realimganalysis.png", add_grid=True, noise_level=10) # Less noise, with grid

    print("\n--- Analyzing 'Real-Like' Image ---")
    real_results = analyze_image_fft("./realimganalysis.png", display_plots=True, peak_prominence_threshold=0.15)
    if real_results:
        if real_results['has_suspicious_peaks']:
            print("Verdict for real-like: Potentially suspicious (could be natural texture or false positive).")
        else:
            print("Verdict for real-like: Appears to have a natural frequency spectrum.")

    print("\n--- Analyzing 'AI-Like' Image with Artifact ---")
    ai_results = analyze_image_fft("./realimganalysis.png", display_plots=True, peak_prominence_threshold=0.15)
    if ai_results:
        if ai_results['has_suspicious_peaks']:
            print("Verdict for AI-like: Detected suspicious peaks, potentially AI-generated or strong periodic pattern.")
        else:
            print("Verdict for AI-like: No clear suspicious peaks detected (AI might be very good or artifacts are subtle).")
    
    # Example with a real image (you need to provide a path)
    # my_image_path = "path/to/your/image.jpg" 
    # if os.path.exists(my_image_path):
    #     print(f"\n--- Analyzing your image: {my_image_path} ---")
    #     custom_results = analyze_image_fft(my_image_path, display_plots=True)
    #     # Further interpretation based on custom_results
    # else:
    #     print(f"\nSkipping custom image analysis, path not found: {my_image_path}")