# AirTist - Air Canvas Drawing Using Hand Gestures

**AirTist** is an advanced virtual drawing application that uses hand tracking technology to draw on a digital canvas. Leveraging OpenCV and MediaPipe, it enables users to draw, erase, and perform other functions with simple hand gestures.

## ✨ Features

- 🎨 **Draw with Hand Gestures**: Use your index finger to draw on the virtual canvas.
- 🧽 **Eraser Tool**: Easily switch to eraser mode for corrections.
- 💾 **Save Your Artwork**: Capture and save your drawings as image files.
- ↩️ **Undo/Redo**: Seamlessly undo and redo your actions for better control.
- 🔍 **Smooth Gesture Tracking**: Enhanced accuracy with a smoothing algorithm.
- 🎨 **Color Selection**: Choose between multiple colors to create vibrant designs.
- ✋ **Multi-Hand Support**: Supports up to two hands for more interactive features.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## 📂 Project Structure

```
├── AirTist
│   ├── airtist.py      # Main application file
│   ├── README.md       # Project documentation
│   └── requirements.txt # Required libraries
```

## 🚀 How to Run

1. **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/AirTist.git
    cd AirTist
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # For Linux/macOS
    .\venv\Scripts\activate    # For Windows
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```bash
    python airtist.py
    ```

## 📸 Usage Instructions

- **Draw**: Pinch your thumb and index finger together to start drawing.
- **Erase**: Select the "Eraser" button and pinch to erase parts of the drawing.
- **Change Colors**: Tap on any color button to switch between pink, green, red, and blue.
- **Clear**: Use the "Clear" button to erase the entire canvas.
- **Save**: Capture your artwork by pressing the "Save" button.
- **Undo/Redo**: Press 'u' to undo and 'r' to redo your previous actions.

## 📌 Keyboard Shortcuts

- Press **ESC**: Exit the application
- Press **U**: Undo last action
- Press **R**: Redo last undone action

## 🧑‍💻 Contribution

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## 📃 License

This project is licensed under the MIT License.

## 📧 Contact

For any inquiries, feel free to reach out via GitHub issues.

---

Enjoy drawing in the air with **AirTist**! 🎨✋

