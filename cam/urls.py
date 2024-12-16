from .views import *
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', index, name='index'),
    path('live-stream/', live_stream, name='live_stream'),
    #path('video_stream/', VideoStreamView.as_view(), name='video_stream'),
    #path('video_feed/', VideoFeedView.as_view(), name='video_feed'),
    #path('select_video/', VideoSelectionView.as_view(), name='select_video'),
    #path('display_video/', VideoDisplayView.as_view(), name='display_video'),
    path('stream/', stream_flv, name='stream_flv'),
    path('stream-view/', view_stream, name='stream_view'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
