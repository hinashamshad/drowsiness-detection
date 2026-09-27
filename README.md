\# 🚗 Drowsiness Detection System



Real-time driver drowsiness detection using Computer Vision. Alerts the driver when signs of fatigue are detected.



\## ✨ Features



\- 👁️ \*\*Eye Aspect Ratio (EAR)\*\* based drowsiness detection

\- 🔔 \*\*Audio alarm\*\* when eyes closed for 2+ seconds

\- 🥱 \*\*Yawn detection\*\* (MAR - Mouth Aspect Ratio)

\- 📊 \*\*Real-time FPS counter\*\*

\- ⚡ Works on CPU at 60+ FPS

\- 🛡️ No external hardware required



\## 🛠️ Tech Stack



| Technology | Purpose |

|-----------|---------|

| Python 3.12 | Core language |

| OpenCV | Webcam \& visualization |

| MediaPipe FaceMesh | 468 face landmarks |

| NumPy | EAR/MAR calculations |

| Winsound | Audio alarm |



\## 🚀 Installation



```bash

git clone https://github.com/hinashamshad/drowsiness-detection.git

cd drowsiness-detection

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

python drowsiness\_detector.py

