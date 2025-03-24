#!/bin/bash

# This script prepares the static site for deployment to GitHub Pages or Vercel

# Create a temporary directory for deployment
mkdir -p deploy_temp

# Copy the static site files to the deployment directory
cp -r static_site/* deploy_temp/

# If deploying to the root of the repository, you can copy files to the repo root
# Uncomment these lines to copy to repository root
# cp -r static_site/* .

echo "Static site files are ready for deployment in the deploy_temp directory."
echo "To deploy to GitHub Pages or Vercel, copy these files to your repository."
echo ""
echo "For GitHub Pages (AI-Bobby-Newb.github.io):"
echo "1. Copy all files from deploy_temp/ to the root of your repository"
echo "2. Push to GitHub"
echo "3. Your site will be available at https://AI-Bobby-Newb.github.io"
echo ""
echo "For Vercel:"
echo "1. Copy all files from deploy_temp/ to the root of your repository"
echo "2. Push to GitHub"
echo "3. Connect to Vercel and deploy"