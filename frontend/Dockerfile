FROM nginx:alpine

# Copy static files to nginx html directory
COPY static/ /usr/share/nginx/html/static/
COPY static/index.html /usr/share/nginx/html/index.html

# Expose port 80
EXPOSE 80

# Standard startup for nginx but we inject the BACKEND_URL environment variable
# into a config.js file that the frontend can read.
CMD ["sh", "-c", "echo \"window.BACKEND_URL = '$BACKEND_URL';\" > /usr/share/nginx/html/config.js && exec nginx -g 'daemon off;'"]
