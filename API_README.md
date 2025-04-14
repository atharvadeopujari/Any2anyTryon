# Any2anyTryon API

This API allows you to integrate the Any2anyTryon virtual try-on functionality into your own website. The API is built with Flask and provides endpoints to process model and garment images.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for optimal performance)
- Required Python packages (install from requirements.txt)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Any2anyTryon.git
   cd Any2anyTryon
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional dependencies for the API:
   ```
   pip install flask flask-cors
   ```

## Running the API

1. Start the API server:
   ```
   python api.py --host 0.0.0.0 --port 5000
   ```

   Options:
   - `--host`: Host to bind the server to (default: 0.0.0.0)
   - `--port`: Port to bind the server to (default: 5000)
   - `--group_offloading`: Enable group offloading for memory optimization

2. The API will be available at `http://localhost:5000/` (or the host/port you specified)

## API Endpoints

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: JSON object confirming the API is running
  ```json
  {
    "status": "ok",
    "message": "API is running"
  }
  ```

### Virtual Try-On

- **URL**: `/api/tryon`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `model_image` (file, optional): Image of the person/model
  - `garment_image` (file, optional): Image of the garment
  - `prompt` (string, optional): Text prompt for generation (default: `<MODEL> a person with fashion garment. <GARMENT> a garment. <TARGET> model with fashion garment`)
  - `height` (integer, optional): Height of the output image (default: 576)
  - `width` (integer, optional): Width of the output image (default: 576)
  - `seed` (integer, optional): Random seed for reproducibility (default: 0)
  - `guidance_scale` (float, optional): Guidance scale for generation (default: 3.5)
  - `num_inference_steps` (integer, optional): Number of inference steps (default: 15)

- **Response**: JSON object containing base64-encoded generated image
  ```json
  {
    "status": "success",
    "image": "base64_encoded_image_data"
  }
  ```

- **Error Response**:
  ```json
  {
    "error": "Error message"
  }
  ```

## Integrating with Your Website

1. Include the API endpoint in your frontend code:

```javascript
async function generateTryOn(modelImage, garmentImage, params = {}) {
  const formData = new FormData();
  
  if (modelImage) {
    formData.append('model_image', modelImage);
  }
  
  if (garmentImage) {
    formData.append('garment_image', garmentImage);
  }
  
  // Add optional parameters
  if (params.prompt) formData.append('prompt', params.prompt);
  if (params.height) formData.append('height', params.height);
  if (params.width) formData.append('width', params.width);
  if (params.seed) formData.append('seed', params.seed);
  if (params.guidance_scale) formData.append('guidance_scale', params.guidance_scale);
  if (params.num_inference_steps) formData.append('num_inference_steps', params.num_inference_steps);
  
  try {
    const response = await fetch('http://your-api-url/api/tryon', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Unknown error occurred');
    }
    
    return data.image; // Base64 encoded image
  } catch (error) {
    console.error('Error generating try-on image:', error);
    throw error;
  }
}
```

2. Display the generated image:

```javascript
// Example usage in your application
const modelImageFile = document.getElementById('modelInput').files[0];
const garmentImageFile = document.getElementById('garmentInput').files[0];

generateTryOn(modelImageFile, garmentImageFile, {
  prompt: '<MODEL> a person with fashion garment. <GARMENT> a garment. <TARGET> model with fashion garment',
  height: 576,
  width: 576,
  num_inference_steps: 15
})
  .then(base64Image => {
    // Display the image
    document.getElementById('resultImage').src = 'data:image/png;base64,' + base64Image;
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

## Example Implementation

A complete HTML example is provided in the `website_example.html` file.

## Production Deployment Considerations

For production deployment, consider the following:

1. **Use a production WSGI server** like Gunicorn or uWSGI instead of Flask's development server:
   ```
   pip install gunicorn
   gunicorn -w 1 -b 0.0.0.0:5000 api:app
   ```

2. **Set up a reverse proxy** with Nginx or Apache to handle SSL termination and load balancing.

3. **Add authentication** to protect your API from unauthorized use.

4. **Implement rate limiting** to prevent abuse.

5. **Deploy on a machine with a powerful GPU** for optimal performance.

6. **Use environment variables** for configuration instead of hardcoded values.

## Troubleshooting

- **CUDA out of memory errors**: Reduce the batch size, image dimensions, or number of inference steps. You can also try enabling the `--group_offloading` option.
- **Slow performance**: Ensure you're running on a machine with a compatible CUDA GPU.
- **CORS errors**: Make sure your frontend domain is allowed by the API's CORS settings.

## License

This project is licensed under the terms of the original Any2anyTryon repository's license. 