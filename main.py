import numpy as np
import os
from PIL import Image

class EnhancedNightVisionProcessor:
    # ==================== NOISE MINIMIZATION MODULE ====================

    def median_filter(self, image, kernel_size=3):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd for symmetric neighborhood")

        # Handle both grayscale and color images
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for c in range(image.shape[2]):
                filtered[:, :, c] = self._apply_median_filter_2d(image[:, :, c], kernel_size)
            return filtered
        else:
            return self._apply_median_filter_2d(image, kernel_size)

    def _apply_median_filter_2d(self, image, kernel_size):
        """Apply median filter to a 2D grayscale image"""
        height, width = image.shape
        pad_size = kernel_size // 2

        # Pad the image to handle boundaries
        padded = np.pad(image, pad_size, mode='edge')
        filtered = np.zeros_like(image)

        # Slide the window across the entire image
        for i in range(height):
            for j in range(width):
                # Extract the neighborhood window
                window = padded[i:i+kernel_size, j:j+kernel_size]
                # Find median
                filtered[i, j] = np.median(window)

        return filtered.astype(np.uint8)

    def gaussian_filter(self, image, sigma=1.0):
        # Create Gaussian kernel
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self._create_gaussian_kernel(kernel_size, sigma)

        # Apply convolution
        if len(image.shape) == 3:
            filtered = np.zeros_like(image)
            for c in range(image.shape[2]):
                filtered[:, :, c] = self._convolve_2d(image[:, :, c], kernel)
            return filtered
        else:
            return self._convolve_2d(image, kernel)

    def _create_gaussian_kernel(self, size, sigma):
        """Create a 2D Gaussian kernel from mathematical formula"""
        center = size // 2
        x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)

        # Apply Gaussian formula
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalize kernel so all values sum to 1
        kernel = kernel / np.sum(kernel)

        return kernel

    def _convolve_2d(self, image, kernel):
        """2D Convolution Implementation from Scratch"""
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i+kernel_height, j:j+kernel_width]
                output[i, j] = np.sum(region * kernel)

        return output.astype(np.uint8)

    # ==================== CONTRAST ENHANCEMENT MODULE ====================

    def histogram_equalization(self, image):
        if len(image.shape) == 3:
            equalized = np.zeros_like(image)
            for c in range(image.shape[2]):
                equalized[:, :, c] = self._equalize_channel(image[:, :, c])
            return equalized
        else:
            return self._equalize_channel(image)

    def _equalize_channel(self, channel):
        """Apply histogram equalization to a single channel"""
        if channel.dtype != np.uint8:
            channel = (channel * 255).astype(np.uint8)

        # Calculate histogram
        histogram = np.zeros(256, dtype=int)
        height, width = channel.shape
        total_pixels = height * width

        for i in range(height):
            for j in range(width):
                histogram[channel[i, j]] += 1

        # Calculate CDF
        cdf = np.zeros(256)
        cdf[0] = histogram[0] / total_pixels

        for i in range(1, 256):
            cdf[i] = cdf[i-1] + histogram[i] / total_pixels

        # Create transformation lookup table
        transform_table = np.round(cdf * 255).astype(np.uint8)

        # Apply transformation
        equalized = np.zeros_like(channel)
        for i in range(height):
            for j in range(width):
                equalized[i, j] = transform_table[channel[i, j]]

        return equalized

    # ==================== DETAIL ENHANCEMENT MODULE ====================

    def unsharp_masking(self, image, radius=1.0, amount=1.5, threshold=0):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Create blurred version
        blurred = self.gaussian_filter(image, sigma=radius)

        # Calculate high-pass filter
        original_float = image.astype(np.float64)
        blurred_float = blurred.astype(np.float64)
        high_pass = original_float - blurred_float

        # Apply threshold if specified
        if threshold > 0:
            mask = np.abs(high_pass) >= threshold
            high_pass = high_pass * mask

        # Add scaled details back
        enhanced = original_float + amount * high_pass
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    # ==================== FALSE-COLOR MAPPING MODULE ====================

    def create_night_vision_false_color(self, enhanced_image):
        # Ensure grayscale input
        if len(enhanced_image.shape) == 3:
            grayscale = np.mean(enhanced_image, axis=2).astype(np.uint8)
        else:
            grayscale = enhanced_image.astype(np.uint8)

        height, width = grayscale.shape
        false_color = np.zeros((height, width, 3), dtype=np.uint8)

        # Classic green night vision mapping
        false_color[:, :, 0] = (grayscale * 0.05).astype(np.uint8)  # Minimal red
        false_color[:, :, 1] = grayscale                             # Full green
        false_color[:, :, 2] = (grayscale * 0.1).astype(np.uint8)   # Slight blue

        return false_color

    # ==================== MAIN PROCESSING PIPELINE ====================

    def enhance_image(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            image = np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

        # Convert to grayscale for processing
        if len(image.shape) == 3:
            weights = np.array([0.299, 0.587, 0.114])
            grayscale = np.dot(image[...,:3], weights).astype(np.uint8)
        else:
            grayscale = image

        # Step 1: Noise Reduction
        median_filtered = self.median_filter(grayscale, kernel_size=3)
        denoised = self.gaussian_filter(median_filtered, sigma=0.8)

        # Step 2: Contrast Enhancement
        contrast_enhanced = self.histogram_equalization(denoised)

        # Step 3: Detail Enhancement
        detail_enhanced = self.unsharp_masking(contrast_enhanced, radius=1.0, amount=1.5)

        # Step 4: False-Color Mapping
        night_vision_image = self.create_night_vision_false_color(detail_enhanced)

        return night_vision_image

def save_image(image, output_path):
    """Save enhanced image to file"""
    Image.fromarray(image).save(output_path)
    print(f"✅ Enhanced image saved to: {output_path}")

if __name__ == "__main__":
    input_path = './img/desert.jpg'
    output_dir = 'img'

    filename, ext = os.path.splitext(os.path.basename(input_path))
    out_path = os.path.join(output_dir, f"{filename}_enhanced{ext}")

    processor = EnhancedNightVisionProcessor()

    # Process image
    final_image = processor.enhance_image(input_path)

    # Save result
    save_image(final_image, out_path)