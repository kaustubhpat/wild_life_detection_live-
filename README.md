## Overview
Wild Life Detection is a versatile and powerful tool that extends far beyond basic life detection, offering enhanced awareness and preparedness in a wide range of scenarios.
# Project Structure
```scss
Wild_LifeDetection/
│
├── Camera/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── cam/
|   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   ├── models.py
│   ├── tests.py
│   └── views.py
|   ├──migrations/
|      ├── __init__.py
│
└── manage.py
```
## SetUp
- Python Version 3.10>~ should work, project was created in `Python Version 3.10.12`
### Prerequisites
- Python 3.x installed
- pip (Python package manager) installed
- A virtual environment (optional but recommended)
- Django installed
- Gunicorn installed for production deployment
- RTMP (server side as well as project)
  - ``` sudo apt update
        sudo apt install libnginx-mod-rtmp
        nginx -V | grep rtmp
    ```
- Nginx
- IP WebCam android mobile app
  - Click on Push streaming
    - click on RTMP configuration
      - Enable RTMP streaming
      - host:port`(<IPaddress>:<port>)`
    - Click on 3 dots & click on start server.
## Installation
- Create and activate virtual environment in Project directory
  `python3.10 -m venv venv && source venv/bin/activate`
  `pip install -r requirements.txt & pip install gunicorn`
- Register the App in Settings
- To use the app within the project, you need to register it in the INSTALLED_APPS list in settings.py. Open camera/settings.py and add your app: 
    ```
        INSTALLED_APPS = [ ...,
        '<app-name>',
        ]
    ```
- Create and Apply Migrations
- Run the following commands to create and apply migrations:
```python manage.py makemigrations && python manage.py migrate ```
- After applying migration now you can run the developement server:
    `python manage.py runserver`
- Run the Django project with Gunicorn use below command:
  - gunicorn -w 2 --bind `0.0.0.0:<port>` --timeout 900 camera.wsgi:application


