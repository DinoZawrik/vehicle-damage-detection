import axios from 'axios'
import { ApiResponse } from './types'

const API_BASE_URL = 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
})

export const detectDamage = async (file: File): Promise<ApiResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  const response = await apiClient.post<ApiResponse>('/detect', formData)
  return response.data
}

export const getHealth = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE_URL}/health`)
  return response.data
}

export const getModels = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE_URL}/models`)
  return response.data
}
