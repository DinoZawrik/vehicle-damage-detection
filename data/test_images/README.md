# Test Images Directory

This directory contains test images for the Vehicle Damage Detection System.

## Usage

Place test images in this directory for:
- Testing the system
- Development and debugging
- Performance benchmarking
- Demonstration purposes

## Image Requirements

- **Formats**: JPEG, PNG, BMP
- **Size**: 50MB maximum per file
- **Resolution**: 64x64 minimum, 4096x4096 maximum
- **Content**: Vehicle images with visible damage

## Directory Structure

```
test_images/
├── README.md               # This file
├── vehicles_with_damage/   # Images with various damage types
├── vehicles_clean/         # Reference images without damage
├── batch_test_set/         # Set for batch processing tests
└── sample_images/          # Sample images for quick testing
```

## Naming Convention

Use descriptive names for test images:
- `car_scratch_front_left.jpg`
- `dent_rear_bumper.jpg`
- `multiple_damages_side.jpg`
- `clean_vehicle_reference.jpg`

## Adding Test Images

1. **Ensure proper licensing** for any images you add
2. **Follow naming conventions**
3. **Include metadata** if possible
4. **Test images** before adding to repository

## Test Categories

### Damage Types
- `scratch` - Surface scratches
- `dent` - Small to medium dents
- `crack` - Cracks in glass or body
- `severe` - Major damage
- `multiple` - Multiple damage types

### Vehicle Parts
- `front` - Front bumper/hood
- `rear` - Rear bumper/trunk
- `side` - Side panels/doors
- `glass` - Windows/windshield

### Severity Levels
- `minor` - Small, cosmetic damage
- `moderate` - Noticeable but repairable
- `severe` - Significant damage
- `critical` - Severe structural damage

## Example Images

The repository includes sample images demonstrating:
1. **Single damage type** - One clear damage instance
2. **Multiple damages** - Several damages on same vehicle
3. **Different severities** - Minor to critical damage
4. **Various angles** - Different vehicle perspectives
5. **Lighting conditions** - Different image quality

## Privacy and Licensing

- **Do not include** personally identifiable information
- **Use royalty-free** images or images you own
- **Credit sources** if required
- **Follow copyright laws** in your jurisdiction

## Contributing Test Images

When contributing test images:

1. **Create pull request** with images
2. **Update this README** with descriptions
3. **Ensure images meet** technical requirements
4. **Include metadata** about damage types and severity
5. **Test the system** with new images

## Automated Testing

Test images are used in:
- **Unit tests** - Validating image processing
- **Integration tests** - Testing API endpoints
- **Performance tests** - Benchmarking processing time
- **Regression tests** - Ensuring system stability

## Quality Guidelines

### Image Quality
- **Good lighting** - Clear, well-lit images
- **Sharp focus** - Avoid blurry images
- **Appropriate angle** - Show damage clearly
- **Minimal obstructions** - No objects blocking damage

### Damage Visibility
- **Clear definition** - Damage should be easily visible
- **Appropriate distance** - Close enough to see details
- **Multiple angles** - For complex damages
- **Reference images** - Clean vehicles for comparison

## Performance Notes

- **Larger images** take longer to process
- **High resolution** improves detection accuracy
- **Batch processing** is more efficient
- **Caching** helps with repeated analyses

## Troubleshooting

### Common Issues
1. **"Invalid image format"** - Check file extension
2. **"File too large"** - Reduce image size
3. **"Could not read image"** - Check file corruption
4. **Low confidence scores** - Ensure good image quality

### Best Practices
- **Preprocess images** if necessary
- **Use consistent lighting** across test sets
- **Include both positive and negative examples**
- **Document expected results** for each image

---

For more information, see the main [README.md](../README.md) or [API Documentation](../docs/API_DOCUMENTATION.md).