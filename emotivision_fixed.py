import os
import json
import time
import csv
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from fer.fer import FER


# ============================================================
# EMOTIVISION - REAL-TIME FACIAL RECOGNITION SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "face_data")
MODEL_FILE = os.path.join(BASE_DIR, "emotivision_model.yml")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
CASCADE_FILE = os.path.join(
    BASE_DIR, "haarcascade_frontalface_default.xml"
)

SAMPLES_PER_PERSON = 30
UNKNOWN_THRESHOLD = 70.0
FACE_SIZE = (200, 200)


# ============================================================
# FOUR ALGORITHMS: QUALITATIVE COMPARISON
# ============================================================

ALGORITHMS = [
    {
        "Algorithm": "Eigenfaces (PCA)",
        "Principle": "Projects aligned face images into principal components.",
        "Advantages": "Simple baseline; relatively lightweight.",
        "Limitations": "Sensitive to lighting, alignment and appearance changes.",
        "Complexity": "Low-moderate",
        "Hardware": "CPU",
        "Scalability": "Small, controlled galleries",
        "Status": "Comparison only"
    },
    {
        "Algorithm": "Fisherfaces (LDA)",
        "Principle": "Learns projections that separate labelled identity classes.",
        "Advantages": "Uses class labels to improve discrimination.",
        "Limitations": "Needs representative samples; struggles with changing conditions.",
        "Complexity": "Moderate training",
        "Hardware": "CPU",
        "Scalability": "Retraining may be needed for new identities",
        "Status": "Comparison only"
    },
    {
        "Algorithm": "LBPH",
        "Principle": "Compares local binary texture patterns in grayscale faces.",
        "Advantages": "Lightweight, interpretable and easy to demonstrate.",
        "Limitations": "Sensitive to pose, lighting and image quality.",
        "Complexity": "Low for small galleries",
        "Hardware": "Laptop CPU/webcam",
        "Scalability": "Small demonstration galleries",
        "Status": "Implemented for identity"
    },
    {
        "Algorithm": "Deep Face Embeddings",
        "Principle": "A neural network maps a face to a numerical feature vector.",
        "Advantages": "Can handle variation when trained appropriately.",
        "Limitations": "Model size, bias, privacy and computation require consideration.",
        "Complexity": "Moderate-high",
        "Hardware": "CPU/GPU depends on model",
        "Scalability": "Embeddings can be indexed for larger galleries",
        "Status": "Comparison only"
    }
]


# ============================================================
# MAIN APPLICATION
# ============================================================

class EmotiVision:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "EmotiVision | Face Recognition & Expression Demo"
        )

        self.root.geometry("1180x790")
        self.root.minsize(900, 650)
        self.root.configure(bg="#101522")

        os.makedirs(DATA_DIR, exist_ok=True)

        self.camera = None
        self.camera_running = False
        self.registering = False

        self.registration_count = 0
        self.registration_frames = 0

        self.person_name = ""
        self.person_folder = ""

        self.model_ready = False
        self._emotion_error_reported = False

        self.labels = self.load_labels()

        # LBPH face recognition model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # FER internally expects OpenCV's frontal-face Haar cascade to exist
        # inside cv2.data.haarcascades. Some OpenCV 5 / Python 3.14 installs
        # omit that XML file, so restore it from the copy beside this script.
        self.ensure_fer_cascade()

        # Facial expression detection model
        self.emotion_detector = FER(mtcnn=False)

        # Face detector
        self.face_detector = self.load_face_detector()

        self.create_interface()
        self.refresh_people()

        self.status.set("Ready. Start the camera to begin.")

        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Load previously trained model, if available
        if os.path.exists(MODEL_FILE):
            try:
                self.recognizer.read(MODEL_FILE)
                self.model_ready = True
                self.status.set(
                    "Saved LBPH model loaded. Start the camera."
                )
            except cv2.error:
                self.model_ready = False

    # ========================================================
    # ENSURE FER'S CASCADE FILE EXISTS
    # ========================================================

    def ensure_fer_cascade(self):
        """Make the Haar cascade available at the path FER expects."""
        try:
            target_dir = cv2.data.haarcascades
        except AttributeError:
            target_dir = os.path.join(os.path.dirname(cv2.__file__), "data")

        target_path = os.path.join(
            target_dir, "haarcascade_frontalface_default.xml"
        )

        # If OpenCV already has a valid copy, leave it alone.
        if os.path.isfile(target_path):
            existing = cv2.CascadeClassifier(target_path)
            if not existing.empty():
                return

        # The user downloads/stores this XML beside emotivision.py.
        if os.path.isfile(CASCADE_FILE):
            try:
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(CASCADE_FILE, target_path)
                check = cv2.CascadeClassifier(target_path)
                if not check.empty():
                    print("FER Haar cascade installed:", target_path)
                    return
            except (OSError, cv2.error) as error:
                print("Could not copy FER Haar cascade:", repr(error))

        # Fail with an actionable message rather than a flood of detectMultiScale errors.
        raise FileNotFoundError(
            "FER needs haarcascade_frontalface_default.xml. Place that XML file "
            "in the same folder as emotivision.py and restart the application."
        )

    # ========================================================
    # LOAD FACE DETECTOR
    # ========================================================

    def load_face_detector(self):

        candidates = [CASCADE_FILE]

        try:
            candidates.append(
                os.path.join(
                    cv2.data.haarcascades,
                    "haarcascade_frontalface_default.xml"
                )
            )
        except AttributeError:
            pass

        for path in candidates:

            if os.path.exists(path):

                detector = cv2.CascadeClassifier(path)

                if not detector.empty():
                    print("Face detector loaded:", path)
                    return detector

        messagebox.showerror(
            "Face Detector Missing",
            "Could not load haarcascade_frontalface_default.xml.\n\n"
            "Download it from:\n"
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml\n\n"
            "Save the file beside emotivision.py and restart."
        )

        raise FileNotFoundError(
            "Haar Cascade XML file missing or invalid."
        )

    # ========================================================
    # CREATE MAIN INTERFACE
    # ========================================================

    def create_interface(self):

        header = tk.Frame(
            self.root,
            bg="#101522"
        )

        header.pack(
            fill="x",
            padx=20,
            pady=(15, 8)
        )

        tk.Label(
            header,
            text="EmotiVision",
            font=("Arial", 25, "bold"),
            fg="#b9a2ff",
            bg="#101522"
        ).pack(anchor="w")

        tk.Label(
            header,
            text="REAL-TIME FACE RECOGNITION + FACIAL EXPRESSION DEMO",
            font=("Arial", 10, "bold"),
            fg="#dbe5f5",
            bg="#101522"
        ).pack(anchor="w")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TNotebook",
            background="#101522",
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            padding=(14, 9),
            font=("Arial", 10, "bold"),
            background="#dce6f2",
            foreground="#182235"
        )

        style.map(
            "TNotebook.Tab",
            background=[("selected", "#6d28d9")],
            foreground=[("selected", "#ffffff")]
        )

        self.tabs = ttk.Notebook(self.root)

        self.tabs.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=5
        )

        self.live_tab = tk.Frame(
            self.tabs,
            bg="#101522"
        )

        self.assessment_tab = tk.Frame(
            self.tabs,
            bg="#101522"
        )

        self.tabs.add(
            self.live_tab,
            text=" Live Recognition "
        )

        self.tabs.add(
            self.assessment_tab,
            text=" Algorithm Assessment "
        )

        self.create_live_tab()
        self.create_assessment_tab()

        footer = tk.Frame(
            self.root,
            bg="#202b40"
        )

        footer.pack(
            fill="x",
            side="bottom"
        )

        self.status = tk.StringVar()

        tk.Label(
            footer,
            textvariable=self.status,
            anchor="w",
            font=("Arial", 10),
            fg="#ffffff",
            bg="#202b40",
            padx=15,
            pady=10
        ).pack(fill="x")

    # ========================================================
    # BUTTON DESIGN - HIGH CONTRAST COLOURS
    # ========================================================

    def button(self, parent, text, command, bg):

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#ffffff",
            activebackground=bg,
            activeforeground="#ffffff",
            disabledforeground="#ffffff",
            font=("Arial", 10, "bold"),
            relief="flat",
            cursor="hand2",
            pady=10,
            bd=0,
            highlightthickness=0
        )

    # ========================================================
    # LIVE RECOGNITION TAB
    # ========================================================

    def create_live_tab(self):

        body = tk.Frame(
            self.live_tab,
            bg="#101522"
        )

        body.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12
        )

        panel = tk.Frame(
            body,
            bg="#1c273b",
            padx=10,
            pady=10
        )

        panel.pack(
            side="left",
            fill="both",
            expand=True
        )

        tk.Label(
            panel,
            text="LIVE CAMERA",
            bg="#1c273b",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w",
            pady=(0, 7)
        )

        self.video = tk.Label(
            panel,
            text="Camera stopped\n\nClick Start Camera",
            bg="#080d19",
            fg="#d8e2f0",
            font=("Arial", 15)
        )

        self.video.pack(
            fill="both",
            expand=True
        )

        side = tk.Frame(
            body,
            bg="#101522",
            width=255
        )

        side.pack(
            side="right",
            fill="y",
            padx=(12, 0)
        )

        side.pack_propagate(False)

        tk.Label(
            side,
            text="CONTROLS",
            bg="#101522",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        ).pack(
            anchor="w",
            pady=(0, 6)
        )

        # Each button uses a visible background and white text
        buttons = [
            (
                "Start Camera",
                self.start_camera,
                "#2563eb"
            ),
            (
                "Stop Camera",
                self.stop_camera,
                "#475569"
            ),
            (
                "Register Face",
                self.register_face,
                "#047857"
            ),
            (
                "Train LBPH Model",
                self.train_model,
                "#6d28d9"
            ),
            (
                "Reload Saved Model",
                self.reload_model,
                "#0e7490"
            )
        ]

        for text, command, color in buttons:

            self.button(
                side,
                text,
                command,
                color
            ).pack(
                fill="x",
                pady=4
            )

        tk.Label(
            side,
            text="REGISTERED PEOPLE",
            bg="#101522",
            fg="#dbe5f5",
            font=("Arial", 10, "bold")
        ).pack(
            anchor="w",
            pady=(15, 5)
        )

        self.people_list = tk.Listbox(
            side,
            height=6,
            bg="#1c273b",
            fg="#ffffff",
            selectbackground="#6d28d9",
            selectforeground="#ffffff",
            relief="flat",
            highlightthickness=0
        )

        self.people_list.pack(fill="x")

        workflow = (
            "WORKFLOW\n\n"
            "1. Start camera\n"
            "2. Register each person\n"
            "3. Train the LBPH model\n"
            "4. Test identity and expression labels\n\n"
            "Expression labels are estimates, not proof "
            "of how a person feels."
        )

        tk.Label(
            side,
            text=workflow,
            justify="left",
            wraplength=235,
            bg="#101522",
            fg="#dbe5f5",
            font=("Arial", 9)
        ).pack(
            anchor="w",
            pady=14
        )

    # ========================================================
    # ALGORITHM ASSESSMENT TAB
    # ========================================================

    def create_assessment_tab(self):

        panel = tk.Frame(
            self.assessment_tab,
            bg="#101522"
        )

        panel.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12
        )

        tk.Label(
            panel,
            text="FOUR-ALGORITHM ASSESSMENT",
            bg="#101522",
            fg="#b9a2ff",
            font=("Arial", 16, "bold")
        ).pack(
            anchor="w",
            pady=5
        )

        tk.Label(
            panel,
            text=(
                "Qualitative comparison only; these are not "
                "measured accuracy results."
            ),
            bg="#101522",
            fg="#dbe5f5"
        ).pack(
            anchor="w",
            pady=(0, 8)
        )

        columns = (
            "Algorithm",
            "Advantages",
            "Limitations",
            "Complexity",
            "Hardware",
            "Status"
        )

        self.table = ttk.Treeview(
            panel,
            columns=columns,
            show="headings",
            height=9
        )

        widths = {
            "Algorithm": 145,
            "Advantages": 190,
            "Limitations": 220,
            "Complexity": 125,
            "Hardware": 150,
            "Status": 160
        }

        for column in columns:

            self.table.heading(
                column,
                text=column
            )

            self.table.column(
                column,
                width=widths[column],
                minwidth=90,
                anchor="w"
            )

        for algorithm in ALGORITHMS:

            self.table.insert(
                "",
                "end",
                values=tuple(
                    algorithm[column]
                    for column in columns
                )
            )

        self.table.pack(
            fill="both",
            expand=True,
            pady=8
        )

        tk.Label(
            panel,
            text="Evaluation Criteria",
            bg="#101522",
            fg="#ffffff",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        criteria = (
            "Accuracy: test on separate images not used for training.\n"
            "Complexity: record training and inference time.\n"
            "Scalability: test increasing numbers of identities.\n"
            "Flexibility: test lighting, pose, distance and occlusion.\n"
            "Privacy: obtain consent and protect or delete stored face images."
        )

        tk.Label(
            panel,
            text=criteria,
            bg="#101522",
            fg="#dbe5f5",
            justify="left"
        ).pack(
            anchor="w",
            pady=5
        )

        row = tk.Frame(
            panel,
            bg="#101522"
        )

        row.pack(
            fill="x",
            pady=8
        )

        self.button(
            row,
            "Export Assessment Report",
            self.export_report,
            "#2563eb"
        ).pack(side="left")

        self.button(
            row,
            "Export Comparison CSV",
            self.export_csv,
            "#475569"
        ).pack(
            side="left",
            padx=8
        )

    # ========================================================
    # LOAD AND SAVE REGISTERED NAMES
    # ========================================================

    def load_labels(self):

        if os.path.exists(LABELS_FILE):

            try:

                with open(
                    LABELS_FILE,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                return {
                    int(key): str(value)
                    for key, value in data.items()
                }

            except (
                OSError,
                ValueError,
                TypeError
            ):
                pass

        return {}

    def save_labels(self):

        with open(
            LABELS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    str(key): value
                    for key, value in self.labels.items()
                },
                file,
                indent=2
            )

    def refresh_people(self):

        self.people_list.delete(
            0,
            tk.END
        )

        for key in sorted(self.labels):

            self.people_list.insert(
                tk.END,
                self.labels[key]
            )

    # ========================================================
    # START CAMERA
    # ========================================================

    def start_camera(self):

        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            self.camera.release()
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Cannot open webcam.\n\n"
                "Close other applications using the camera, "
                "check macOS camera permissions, or try camera index 1."
            )

            return

        self.camera_running = True

        self.status.set("Camera running.")

        self.update_frame()

    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop_camera(self):

        self.camera_running = False
        self.registering = False

        if self.camera is not None:

            self.camera.release()
            self.camera = None

        self.video.configure(
            image="",
            text="Camera stopped\n\nClick Start Camera"
        )

        self.video.image = None

        self.status.set("Camera stopped.")

    # ========================================================
    # REGISTER A PERSON
    # ========================================================

    def register_face(self):

        if not self.camera_running:

            messagebox.showinfo(
                "Start Camera",
                "Start the camera before registering a face."
            )

            return

        name = simpledialog.askstring(
            "Register Face",
            "Enter the person's name:",
            parent=self.root
        )

        if not name or not name.strip():
            return

        name = "".join(
            character
            for character in name.strip()
            if character.isalnum()
            or character in (" ", "_", "-")
        ).strip()

        if not name:

            messagebox.showerror(
                "Invalid Name",
                "Enter a valid name."
            )

            return

        existing_id = next(
            (
                key
                for key, value in self.labels.items()
                if value.casefold() == name.casefold()
            ),
            None
        )

        if existing_id is not None:

            self.person_id = existing_id

        else:

            self.person_id = (
                max(self.labels.keys(), default=-1) + 1
            )

        self.labels[self.person_id] = name

        self.person_name = name

        self.person_folder = os.path.join(
            DATA_DIR,
            name
        )

        os.makedirs(
            self.person_folder,
            exist_ok=True
        )

        self.registration_count = 0
        self.registration_frames = 0
        self.registering = True

        self.save_labels()
        self.refresh_people()

        self.status.set(
            f"Registering {name}. Look at the camera "
            "and slowly turn your head."
        )

    # ========================================================
    # TRAIN LBPH MODEL
    # ========================================================

    def train_model(self):

        faces = []
        ids = []

        for person_id, name in self.labels.items():

            folder = os.path.join(
                DATA_DIR,
                name
            )

            if not os.path.isdir(folder):
                continue

            for filename in os.listdir(folder):

                if not filename.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                path = os.path.join(
                    folder,
                    filename
                )

                image = cv2.imread(
                    path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is not None:

                    faces.append(
                        cv2.resize(image, FACE_SIZE)
                    )

                    ids.append(person_id)

        if len(faces) < 2:

            messagebox.showwarning(
                "Insufficient Data",
                "Register a person and collect face samples first."
            )

            return

        try:

            self.recognizer = (
                cv2.face.LBPHFaceRecognizer_create()
            )

            self.recognizer.train(
                faces,
                np.asarray(ids, dtype=np.int32)
            )

            self.recognizer.save(MODEL_FILE)

            self.model_ready = True

            messagebox.showinfo(
                "Training Complete",
                f"Trained with {len(faces)} images from "
                f"{len(set(ids))} identity/identities."
            )

            self.status.set(
                f"LBPH model trained with {len(faces)} samples."
            )

        except cv2.error as error:

            messagebox.showerror(
                "Training Error",
                str(error)
            )

    # ========================================================
    # RELOAD SAVED MODEL
    # ========================================================

    def reload_model(self):

        if not os.path.exists(MODEL_FILE):

            messagebox.showinfo(
                "No Saved Model",
                "Train the model first."
            )

            return

        try:

            self.recognizer = (
                cv2.face.LBPHFaceRecognizer_create()
            )

            self.recognizer.read(MODEL_FILE)

            self.model_ready = True

            self.status.set("Saved model loaded.")

        except cv2.error as error:

            messagebox.showerror(
                "Load Error",
                str(error)
            )

    # ========================================================
    # LIVE FACE AND EXPRESSION DETECTION
    # ========================================================

    def update_frame(self):

        if not self.camera_running or self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:

            self.status.set(
                "Camera opened but no frame was received."
            )

            self.root.after(
                100,
                self.update_frame
            )

            return

        # Mirror the camera image
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(70, 70)
        )

        for (x, y, width, height) in detected_faces:

            face_gray = gray[
                y:y + height,
                x:x + width
            ]

            identity = "Unknown"
            box_color = (60, 200, 255)

            # -----------------------------------------------
            # FACE REGISTRATION
            # -----------------------------------------------

            if self.registering:

                self.registration_frames += 1

                if (
                    self.registration_frames % 5 == 0
                    and self.registration_count < SAMPLES_PER_PERSON
                ):

                    sample = cv2.resize(
                        face_gray,
                        FACE_SIZE
                    )

                    filename = (
                        f"sample_{int(time.time() * 1000)}.jpg"
                    )

                    sample_path = os.path.join(
                        self.person_folder,
                        filename
                    )

                    if cv2.imwrite(sample_path, sample):

                        self.registration_count += 1

                identity = (
                    f"Registering {self.person_name}: "
                    f"{self.registration_count}/{SAMPLES_PER_PERSON}"
                )

                self.status.set(identity)

                if self.registration_count >= SAMPLES_PER_PERSON:

                    self.registering = False

                    self.status.set(
                        f"{self.person_name} registered. "
                        "Register other participants, then train the model."
                    )

            # -----------------------------------------------
            # IDENTITY RECOGNITION
            # -----------------------------------------------

            elif self.model_ready:

                try:

                    resized_face = cv2.resize(
                        face_gray,
                        FACE_SIZE
                    )

                    person_id, distance = (
                        self.recognizer.predict(resized_face)
                    )

                    if (
                        distance <= UNKNOWN_THRESHOLD
                        and person_id in self.labels
                    ):

                        identity = self.labels[person_id]
                        box_color = (60, 220, 100)

                    else:

                        identity = "Unknown"

                except cv2.error:

                    identity = "Recognition error"

            # -----------------------------------------------
            # FACIAL EXPRESSION DETECTION
            # -----------------------------------------------

            emotion_text = "Expression: analysing..."

            try:

                face_bgr = frame[
                    y:y + height,
                    x:x + width
                ]

                face_rgb = cv2.cvtColor(
                    face_bgr,
                    cv2.COLOR_BGR2RGB
                )

                results = self.emotion_detector.detect_emotions(
                    face_rgb
                )

                if results:

                    scores = results[0].get(
                        "emotions",
                        {}
                    )

                    if scores:

                        best_emotion = max(
                            scores,
                            key=scores.get
                        )

                        confidence = (
                            float(scores[best_emotion]) * 100
                        )

                        emotion_text = (
                            f"Expression: {best_emotion.title()} "
                            f"({confidence:.0f}%)"
                        )

                    else:

                        emotion_text = "Expression: not detected"

                else:

                    emotion_text = "Expression: not detected"

            except Exception as error:

                emotion_text = "Expression: unavailable"

                # Avoid printing the same exception dozens of times per second.
                if not self._emotion_error_reported:
                    print("Expression detection error:", repr(error))
                    self._emotion_error_reported = True

            # -----------------------------------------------
            # DRAW IDENTITY AND EXPRESSION ON CAMERA
            # -----------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                box_color,
                2
            )

            cv2.putText(
                frame,
                identity[:45],
                (x, max(25, y - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
                cv2.LINE_AA
            )

            text_y = min(
                frame.shape[0] - 12,
                y + height + 25
            )

            cv2.putText(
                frame,
                emotion_text[:60],
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 210, 70),
                2,
                cv2.LINE_AA
            )

        # -----------------------------------------------
        # DISPLAY CAMERA FRAME IN TKINTER
        # -----------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        image.thumbnail((800, 570))

        photo = ImageTk.PhotoImage(image)

        self.video.configure(
            image=photo,
            text=""
        )

        # Keep a reference so Tkinter doesn't remove the image
        self.video.image = photo

        self.root.after(
            30,
            self.update_frame
        )

    # ========================================================
    # EXPORT ALGORITHM ASSESSMENT REPORT
    # ========================================================

    def export_report(self):

        path = filedialog.asksaveasfilename(
            title="Save Assessment Report",
            defaultextension=".txt",
            initialfile="EmotiVision_Assessment.txt",
            filetypes=[
                ("Text files", "*.txt")
            ]
        )

        if not path:
            return

        lines = [
            "EMOTIVISION: FACIAL RECOGNITION ALGORITHM ASSESSMENT",
            "=" * 58,
            "",
            "AIM",
            "Build a real-time face recognition prototype and assess four algorithm families.",
            "",
            "IMPLEMENTATION",
            "Haar Cascade: face detection.",
            "LBPH: identity recognition for registered participants.",
            "FER pretrained model: estimates visible facial-expression categories.",
            "Eigenfaces, Fisherfaces and deep embeddings: comparison only in this application.",
            "",
            "FOUR-ALGORITHM COMPARISON"
        ]

        for algorithm in ALGORITHMS:

            lines.extend([
                "",
                algorithm["Algorithm"],
                "-" * len(algorithm["Algorithm"])
            ])

            for key, value in algorithm.items():

                if key != "Algorithm":

                    lines.append(
                        f"{key}: {value}"
                    )

        lines.extend([
            "",
            "EXPERIMENTAL EVALUATION PLAN",
            "1. Obtain informed consent from participants.",
            "2. Keep separate test images that were not used during training.",
            "3. Measure identity recognition accuracy on held-out test images.",
            "4. Record false matches and missed matches.",
            "5. Measure inference time on the same computer.",
            "6. Test lighting, pose, distance and partial occlusion.",
            "7. Evaluate expression labels separately; they are not ground-truth feelings.",
            "8. Report actual measurements only; do not invent accuracy percentages.",
            "",
            "LIMITATIONS AND ETHICS",
            "LBPH distance is not a probability of identity.",
            "Expression classifications estimate visible facial patterns and do not prove internal emotional state.",
            "Performance depends on dataset, camera, pose, lighting and model.",
            "This classroom prototype is not suitable for high-stakes decisions.",
            "Obtain consent, limit access to face images, and delete them when no longer needed.",
            "",
            "ORIGINAL PAPERS TO READ AND VERIFY BEFORE CITING",
            "[1] M. Turk and A. Pentland, 'Eigenfaces for Recognition,' Journal of Cognitive Neuroscience, vol. 3, no. 1, pp. 71-86, 1991.",
            "[2] P. N. Belhumeur, J. P. Hespanha, and D. J. Kriegman, 'Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection,' IEEE TPAMI, vol. 19, no. 7, pp. 711-720, 1997.",
            "[3] T. Ahonen, A. Hadid, and M. Pietikäinen, 'Face Description with Local Binary Patterns: Application to Face Recognition,' IEEE TPAMI, vol. 28, no. 12, pp. 2037-2041, 2006.",
            "[4] F. Schroff, D. Kalenichenko, and J. Philbin, 'FaceNet: A Unified Embedding for Face Recognition and Clustering,' CVPR, 2015, pp. 815-823.",
            "Read the original publications and verify details before submitting."
        ])

        try:

            with open(
                path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "\n".join(lines)
                )

            messagebox.showinfo(
                "Export Complete",
                f"Report saved to:\n{path}"
            )

        except OSError as error:

            messagebox.showerror(
                "Export Error",
                str(error)
            )

    # ========================================================
    # EXPORT ALGORITHM COMPARISON AS CSV
    # ========================================================

    def export_csv(self):

        path = filedialog.asksaveasfilename(
            title="Export Comparison Table",
            defaultextension=".csv",
            initialfile="EmotiVision_Comparison.csv",
            filetypes=[
                ("CSV files", "*.csv")
            ]
        )

        if not path:
            return

        try:

            with open(
                path,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(ALGORITHMS[0].keys())
                )

                writer.writeheader()
                writer.writerows(ALGORITHMS)

            messagebox.showinfo(
                "Export Complete",
                f"Comparison CSV saved to:\n{path}"
            )

        except OSError as error:

            messagebox.showerror(
                "Export Error",
                str(error)
            )

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close(self):

        self.camera_running = False

        if self.camera is not None:

            self.camera.release()

        self.root.destroy()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    root = tk.Tk()

    application = EmotiVision(root)

    root.mainloop()