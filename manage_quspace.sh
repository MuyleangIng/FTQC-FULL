#!/bin/bash

SERVICE_NAME=ftqc-backend.service

echo "=== FTQC Backend Service Manager ==="
echo "1. Start"
echo "2. Stop"
echo "3. Restart"
echo "4. Status"
echo "5. View Logs (Live)"
echo "6. View Logs (Today)"
echo "7. View Logs (Last Hour)"
echo "0. Exit"
echo "======================================="

read -p "Choose an option: " OPTION

case $OPTION in
  1) sudo systemctl start $SERVICE_NAME ;;
  2) sudo systemctl stop $SERVICE_NAME ;;
  3) sudo systemctl restart $SERVICE_NAME ;;
  4) sudo systemctl status $SERVICE_NAME ;;
  5) journalctl -u $SERVICE_NAME -f ;;
  6) journalctl -u $SERVICE_NAME --since today ;;
  7) journalctl -u $SERVICE_NAME --since "1 hour ago" ;;
  0) exit 0 ;;
  *) echo "Invalid option!" ;;
esac
