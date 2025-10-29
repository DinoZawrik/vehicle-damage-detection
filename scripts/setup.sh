#!/bin/bash

# Vehicle Damage Detection System - Setup Script
# This script helps you get started quickly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check for required commands
    local missing_commands=()
    
    if ! command_exists docker; then
        missing_commands+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_commands+=("docker-compose")
    fi
    
    if ! command_exists git; then
        missing_commands+=("git")
    fi
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        print_error "Missing required commands: ${missing_commands[*]}"
        print_error "Please install the missing commands and try again."
        exit 1
    fi
    
    print_success "All required commands are available"
}

# Function to check Docker daemon
check_docker() {
    print_status "Checking Docker daemon..."
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running or not accessible"
        print_error "Please start Docker and try again"
        exit 1
    fi
    
    print_success "Docker daemon is running"
}

# Function to create .env file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
            print_warning "Please edit .env file with your configuration"
        else
            print_error ".env.example file not found"
            exit 1
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
}

# Function to create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    local directories=(
        "data/models"
        "data/uploads"
        "data/processed"
        "data/raw"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        fi
    done
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    
    if [ -f docker-compose.yml ]; then
        docker-compose build --no-cache
        print_success "Docker images built successfully"
    else
        print_error "docker-compose.yml not found"
        exit 1
    fi
}

# Function to start services
start_services() {
    print_status "Starting services..."
    
    if [ -f docker-compose.yml ]; then
        docker-compose up -d
        print_success "Services started successfully"
        
        print_status "Waiting for services to be ready..."
        sleep 30
        
        # Check if services are healthy
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "API is healthy and ready"
        else
            print_warning "API health check failed, please check logs with: docker-compose logs -f api"
        fi
        
        if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_success "UI is healthy and ready"
        else
            print_warning "UI health check failed, please check logs with: docker-compose logs -f streamlit"
        fi
    else
        print_error "docker-compose.yml not found"
        exit 1
    fi
}

# Function to display usage information
show_usage_info() {
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo "🎉 Vehicle Damage Detection System is now running!"
    echo ""
    echo "Access Points:"
    echo "  📱 Web UI:        http://localhost:8501"
    echo "  🔧 API:           http://localhost:8000"
    echo "  📚 API Docs:      http://localhost:8000/docs"
    echo "  💾 MinIO Console: http://localhost:9001 (admin/admin)"
    echo ""
    echo "Useful Commands:"
    echo "  View logs:        docker-compose logs -f"
    echo "  Stop services:    docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Clean up:         docker-compose down -v"
    echo ""
    echo "Next Steps:"
    echo "  1. Open http://localhost:8501 in your browser"
    echo "  2. Upload a vehicle image for analysis"
    echo "  3. Explore the API documentation at http://localhost:8000/docs"
    echo ""
}

# Function to show help
show_help() {
    echo "Vehicle Damage Detection System Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --check    Only check requirements"
    echo "  -e, --env      Setup environment only"
    echo "  -d, --dirs     Create directories only"
    echo "  -b, --build    Build Docker images only"
    echo "  -s, --start    Start services only"
    echo "  -a, --all      Full setup (default)"
    echo ""
}

# Main setup function
full_setup() {
    echo "🚀 Vehicle Damage Detection System Setup"
    echo "========================================"
    echo ""
    
    check_requirements
    check_docker
    setup_environment
    setup_directories
    build_images
    start_services
    show_usage_info
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -c|--check)
        check_requirements
        check_docker
        print_success "All checks passed!"
        exit 0
        ;;
    -e|--env)
        setup_environment
        exit 0
        ;;
    -d|--dirs)
        setup_directories
        exit 0
        ;;
    -b|--build)
        check_requirements
        check_docker
        build_images
        exit 0
        ;;
    -s|--start)
        check_requirements
        check_docker
        start_services
        show_usage_info
        exit 0
        ;;
    -a|--all|"")
        full_setup
        exit 0
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac