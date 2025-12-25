#!/bin/sh 
# echo "Launch Browser"
#DISPLAY=:0 midori -p -e Fullscreen -a http://localhost:8080/social/marrtina.html 
#rm /home/marrtino/snap/chromium/common/chromium/SingletonLock 
chromium --app=http:clock.html --kiosk -start-maximized
