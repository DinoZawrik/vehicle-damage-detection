import React, { useState, useCallback } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper, 
  CircularProgress, 
  Alert, 
  Grid, 
  Card, 
  CardContent,
  Chip,
  Stack,
  Divider
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { Upload as UploadIcon, Image as ImageIcon } from '@mui/icons-material';
import { detectDamage } from './api';
import { ApiResponse } from './types';

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const data = await detectDamage(selectedFile);
      setResult(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'image/jpeg': [],
      'image/png': []
    },
    multiple: false
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'minor': return 'success';
      case 'moderate': return 'warning';
      case 'severe': return 'error';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mb: 4, fontWeight: 'bold' }}>
        🚗 Vehicle Damage Detection
      </Typography>
      
      <Grid container spacing={4}>
        {/* Left Column: Upload & Image */}
        <Grid item xs={12} md={8}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 0, 
              overflow: 'hidden', 
              position: 'relative',
              minHeight: 400,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              bgcolor: '#f5f5f5'
            }}
          >
            {!preview ? (
              <Box 
                {...getRootProps()} 
                sx={{ 
                  p: 6, 
                  textAlign: 'center', 
                  cursor: 'pointer',
                  width: '100%',
                  height: '100%',
                  bgcolor: isDragActive ? '#e3f2fd' : 'transparent',
                  transition: 'background-color 0.2s'
                }}
              >
                <input {...getInputProps()} />
                <UploadIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  Drag & drop an image here, or click to select
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Supports JPG, PNG (Max 10MB)
                </Typography>
              </Box>
            ) : (
              <Box sx={{ position: 'relative', width: '100%' }}>
                <img 
                  src={preview} 
                  alt="Preview" 
                  style={{ width: '100%', display: 'block' }} 
                />
                
                {/* Bounding Boxes Overlay */}
                {result && result.detections.map((det, idx) => {
                  const [x1, y1, x2, y2] = det.bbox;
                  const imgWidth = result.image_info.width;
                  const imgHeight = result.image_info.height;
                  
                  return (
                    <Box
                      key={idx}
                      sx={{
                        position: 'absolute',
                        left: `${(x1 / imgWidth) * 100}%`,
                        top: `${(y1 / imgHeight) * 100}%`,
                        width: `${((x2 - x1) / imgWidth) * 100}%`,
                        height: `${((y2 - y1) / imgHeight) * 100}%`,
                        border: '2px solid',
                        borderColor: det.class_name === 'scratch' ? '#ffeb3b' : 
                                   det.class_name === 'dent' ? '#ff9800' : 
                                   det.class_name === 'crack' ? '#f44336' : '#00bcd4',
                        boxShadow: '0 0 4px rgba(0,0,0,0.5)',
                        '&:hover::after': {
                          content: `"${det.class_name} (${Math.round(det.confidence * 100)}%)"`,
                          position: 'absolute',
                          top: -25,
                          left: 0,
                          bgcolor: 'rgba(0,0,0,0.8)',
                          color: 'white',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          whiteSpace: 'nowrap'
                        }
                      }}
                    />
                  );
                })}

                {/* Reset Button Overlay */}
                <Box 
                  sx={{ 
                    position: 'absolute', 
                    top: 16, 
                    right: 16, 
                    zIndex: 10 
                  }}
                >
                  <Box 
                    {...getRootProps()} 
                    sx={{ 
                      bgcolor: 'rgba(255,255,255,0.9)', 
                      p: 1, 
                      borderRadius: 1, 
                      cursor: 'pointer',
                      boxShadow: 2,
                      '&:hover': { bgcolor: 'white' }
                    }}
                  >
                    <input {...getInputProps()} />
                    <Typography variant="button" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <ImageIcon fontSize="small" /> New Image
                    </Typography>
                  </Box>
                </Box>
              </Box>
            )}
            
            {loading && (
              <Box 
                sx={{ 
                  position: 'absolute', 
                  top: 0, 
                  left: 0, 
                  right: 0, 
                  bottom: 0, 
                  bgcolor: 'rgba(255,255,255,0.7)', 
                  display: 'flex', 
                  justifyContent: 'center', 
                  alignItems: 'center',
                  zIndex: 20
                }}
              >
                <CircularProgress size={60} />
              </Box>
            )}
          </Paper>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Grid>

        {/* Right Column: Results */}
        <Grid item xs={12} md={4}>
          <Typography variant="h5" gutterBottom>
            Analysis Report
          </Typography>
          
          {!result ? (
            <Paper sx={{ p: 3, bgcolor: '#f9f9f9', color: 'text.secondary' }}>
              <Typography>
                Upload an image to see the damage analysis report here.
              </Typography>
            </Paper>
          ) : (
            <Stack spacing={3}>
              {/* Summary Card */}
              <Card>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    OVERALL SEVERITY
                  </Typography>
                  <Chip 
                    label={result.analysis.severity.toUpperCase()} 
                    color={getSeverityColor(result.analysis.severity) as any}
                    sx={{ fontWeight: 'bold', px: 1 }}
                  />
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    ESTIMATED COST
                  </Typography>
                  <Typography variant="h4" color="primary.main">
                    ${result.cost_estimate.min} - ${result.cost_estimate.max}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {result.cost_estimate.currency}
                  </Typography>
                </CardContent>
              </Card>

              {/* Details Card */}
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detected Damages
                  </Typography>
                  
                  {result.detections.length === 0 ? (
                    <Typography color="text.secondary">No damages detected.</Typography>
                  ) : (
                    <Stack spacing={1}>
                      {Object.entries(result.analysis.damage_types).map(([type, count]) => (
                        <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography sx={{ textTransform: 'capitalize' }}>{type}</Typography>
                          <Chip label={count} size="small" />
                        </Box>
                      ))}
                    </Stack>
                  )}
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Processing Time</Typography>
                    <Typography variant="body2">{result.processing_time}s</Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">Model</Typography>
                    <Typography variant="body2">{result.model.name}</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Stack>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default App;
