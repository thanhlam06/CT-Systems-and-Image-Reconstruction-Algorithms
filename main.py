"""
main.py - Điểm khởi động chính của ứng dụng CT Simulator Pro.
"""
import sys
import os

# Thêm đường dẫn hiện tại vào sys.path để import ct_sim ổn định
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from gui import main

if __name__ == "__main__":
    main()
