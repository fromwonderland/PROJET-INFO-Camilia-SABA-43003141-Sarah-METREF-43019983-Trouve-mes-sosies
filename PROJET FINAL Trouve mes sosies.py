import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import webbrowser
from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import filedialog
import os
from deepface import DeepFace
import numpy as np
import threading
import time
from datetime import datetime
from fpdf import FPDF
import cv2
import json

# Fonction pour ouvrir le lien YouTube dans le navigateur
def open_help_video():
    webbrowser.open("https://www.youtube.com/watch?v=_S-snes6PPQ", new=2)  # "new=2" ouvre dans un nouvel onglet si possible

# Fonction pour configurer la fenêtre principale
def create_window():
    global root
    # Créer la fenêtre principale
    root = tk.Tk()

    # Définir le titre de la fenêtre
    root.title("🩷Trouve mes sosies🩷")

    # Mettre la fenêtre en presque plein écran
    root.attributes("-fullscreen", True)

    # Définir le fond de la fenêtre en noir
    root.config(bg=bg_color)

    # Créer un bouton "🤔 Aide" dans l'angle supérieur gauche avec taille de police 23px
    help_button = tk.Button(root, text="🤔 Aide", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=open_help_video)
    help_button.place(x=10, y=10)  # Positionner le bouton en haut à gauche (10px de marge)

    # Ajouter un bouton "❌" dans l'angle supérieur droit pour fermer la fenêtre avec taille de police 23px
    close_button = tk.Button(root, text="❌", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=lambda: open_quit_confirmation(root))
    close_button.place(x=root.winfo_screenwidth() - 10, y=10, anchor='ne')  # Positionner le bouton en haut à droite avec une marge de 10px

    # Ajouter un label au centre, légèrement vers le haut
    label = tk.Label(root, text="🩷Entre ton nom et clique sur la fée pour trouver ton sosie🩷", font=("Times New Roman", 24), fg=text_color, bg=bg_color)
    label.pack(pady=(100, 20))  # 100px de marge en haut et 20px de marge en bas

    # Créer un cadre pour l'entrée de texte de 25 caractères
    frame2 = tk.Frame(root, bg=entry_frame_color, bd=1)  # Frame avec bordure #ff82e6, plus fine (bd=1)
    frame2.pack(pady=20)

    # Ajouter l'entrée de texte de 25 caractères dans le cadre, avec une police en gras et texte centré
    entry2 = tk.Entry(frame2, font=("Times New Roman", 20, "bold"), fg=text_color, bg=entry_bg_color, bd=2, relief="solid", width=25, justify='center')
    entry2.pack(padx=10, pady=5)  # Ajouter un peu de marge à l'intérieur du cadre

    # Créer un bouton "⭐" à droite de l'entrée de texte avec taille de police 30px
    def on_star_click():
        # Si l'entrée de texte est vide, changer la couleur du label
        if not entry2.get().strip():  # Si le texte est vide
            label.config(fg="#71CC51")
        else:
            # Si l'entrée de texte contient du texte, fermer la fenêtre actuelle et ouvrir la nouvelle page
            user_name = entry2.get().strip()
            root.destroy()  # Fermer la fenêtre actuelle
            create_etape2_window(user_name)  # Créer la nouvelle fenêtre "Etape 2" avec le nom de l'utilisateur

    star_button = tk.Button(root, text="⭐", font=("Times New Roman", 30), fg=text_color, bg=button_color, command=on_star_click)
    star_button.pack(pady=10)  # Le bouton est centré également

    # Ajouter un bouton "🦋Personnaliser" à gauche du bouton "❌"
    personalise_button = tk.Button(root, text="🦋Personnaliser", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=open_customization_window)
    personalise_button.place(x=root.winfo_screenwidth() - 85, y=10, anchor='ne')  # Positionner à gauche du bouton "❌"

    # Lancer la boucle principale de l'application
    root.mainloop()

# Chemins et configurations
faces_folder = "C:\\Users\\camil\\Music\\Visages"  # Dossier contenant les images

def generate_embeddings_for_folder(new_window):
    """
    Calcule les embeddings pour toutes les images dans le dossier et les renvoie dans un dictionnaire.
    """
    print("Génération des embeddings pour les images du dossier...")

    embeddings = {}
    total_files = len(os.listdir(faces_folder))
    processed_files = 0

    for file_name in os.listdir(faces_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(faces_folder, file_name)
            print(f"Traitement de {file_name}...")
            try:
                # Calcul des embeddings pour l'image
                embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face")[0]["embedding"]
                embeddings[file_name] = embedding
            except Exception as e:
                print(f"Erreur avec {file_name}: {e}")

        processed_files += 1
        progress_percentage = (processed_files / total_files) * 100
        if new_window.winfo_exists():
            progress_var.set(progress_percentage)
            progress_label.config(text=f"Chargement... {int(progress_percentage)}%")

    return embeddings

def find_sosie(user_image, new_window):
    global embeddings, search_thread_active
    # Générer les embeddings pour les images du dossier
    embeddings = generate_embeddings_for_folder(new_window)

    # Calculer l'embedding pour l'image utilisateur
    print("Calcul des embeddings pour l'image utilisateur...")
    try:
        user_embedding = DeepFace.represent(img_path=user_image, model_name="VGG-Face")[0]["embedding"]
    except Exception as e:
        print(f"Erreur lors du traitement de l'image utilisateur : {e}")
        return

    # Trouver le sosie en comparant les embeddings
    best_match = None
    best_distance = float("inf")
    print("Recherche du sosie...")
    distances = []
    max_distance = 0  # Initialiser max_distance à 0

    for file_name, db_embedding in embeddings.items():
        if not search_thread_active:
            return
        # Calculer la distance cosinus entre l'embedding de l'utilisateur et celui de la base de données
        distance = np.linalg.norm(np.array(user_embedding) - np.array(db_embedding))
        print(f"Comparaison avec {file_name} : distance = {distance}")
        distances.append((file_name, distance))

        if distance < best_distance:
            best_distance = distance
            best_match = file_name

        # Mettre à jour max_distance
        if distance > max_distance:
            max_distance = distance

    if best_match:
        print(f"Le sosie le plus proche est {best_match} avec une distance de {best_distance}")
    else:
        print("Aucun sosie trouvé.")

    # Trier les distances par ordre décroissant
    distances.sort(key=lambda x: x[1])

    return best_match, distances, best_distance, max_distance

# Fonction pour ouvrir la fenêtre de confirmation de quitter
def open_quit_confirmation(etape2_window):
    # Créer une nouvelle fenêtre pour la confirmation
    confirmation_window = tk.Toplevel(etape2_window)
    confirmation_window.title("🩷Quitter?🩷")

    # Centrer la fenêtre de confirmation
    screen_width = confirmation_window.winfo_screenwidth()
    screen_height = confirmation_window.winfo_screenheight()
    window_width = 400
    window_height = 200
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    confirmation_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    confirmation_window.config(bg=bg_color)

    # Ajouter un label avec la question
    label = tk.Label(confirmation_window, text="🩷Souhaites-tu quitter l'app?🩷", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
    label.pack(pady=20)

    # Créer un cadre pour les boutons
    button_frame = tk.Frame(confirmation_window, bg=bg_color)
    button_frame.pack(pady=10)

    # Créer les boutons "🩷oui🩷" et "🩷non🩷"
    def quit_app():
        confirmation_window.destroy()  # Fermer la fenêtre de confirmation
        etape2_window.destroy()  # Fermer la fenêtre "Etape 2"

    yes_button = tk.Button(button_frame, text="🩷oui🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=quit_app)
    yes_button.pack(side=tk.LEFT, padx=10)

    no_button = tk.Button(button_frame, text="🩷non🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=confirmation_window.destroy)
    no_button.pack(side=tk.RIGHT, padx=10)

# Fonction pour ouvrir la fenêtre "Historique"
def open_history_window():
    history_window = tk.Toplevel()
    history_window.title("🩷Historique🩷")

    # Centrer la fenêtre "Historique"
    screen_width = history_window.winfo_screenwidth()
    screen_height = history_window.winfo_screenheight()
    window_width = 600
    window_height = 450
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    history_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    history_window.config(bg=bg_color)

    # Ajouter un label pour l'historique
    history_label = tk.Label(history_window, text="🩷Historique de tes sosies🩷", font=("Times New Roman", 20), fg=text_color, bg=bg_color)
    history_label.pack(pady=20)

    # Créer un cadre pour contenir le Canvas et la Scrollbar
    frame = tk.Frame(history_window, bg=bg_color)
    frame.pack(fill=tk.BOTH, expand=1, pady=10)

    # Créer un Canvas pour contenir les widgets
    canvas = tk.Canvas(frame, bg=bg_color, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Ajouter une barre de défilement verticale au Canvas
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configurer le Canvas pour qu'il puisse faire défiler les widgets
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Créer un cadre pour l'historique
    history_frame = tk.Frame(canvas, bg=bg_color)
    canvas.create_window((0, 0), window=history_frame, anchor='nw')

    # Ajouter l'historique à la fenêtre
    for entry in history:
        entry_label = tk.Label(history_frame, text=entry, font=("Times New Roman", 16), fg=text_color, bg=bg_color)
        entry_label.pack(anchor='w')

    # Assurer que le Canvas s'étend correctement
    history_window.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    # Ajouter un bouton pour effacer l'historique dans history_window
    clear_button = tk.Button(history_window, text="🩷Effacer mon historique🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=lambda: open_clear_confirmation(history_window, history_frame))
    clear_button.pack(pady=10)

# Fonction pour ouvrir la fenêtre de confirmation pour effacer l'historique
def open_clear_confirmation(history_window, history_frame):
    confirmation_window = tk.Toplevel(history_window)
    confirmation_window.title("🩷Effacer l'historique des sosies?🩷")

    # Centrer la fenêtre de confirmation
    screen_width = confirmation_window.winfo_screenwidth()
    screen_height = confirmation_window.winfo_screenheight()
    window_width = 400
    window_height = 200
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    confirmation_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    confirmation_window.config(bg=bg_color)

    # Ajouter un label avec la question
    label = tk.Label(confirmation_window, text="🩷Souhaites-tu effacer l'historique de tes sosies?🩷", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
    label.pack(pady=20)

    # Créer un cadre pour les boutons
    button_frame = tk.Frame(confirmation_window, bg=bg_color)
    button_frame.pack(pady=10)

    # Créer les boutons "🩷oui🩷" et "🩷non🩷"
    def clear_history():
        global history
        history = []
        for widget in history_frame.winfo_children():
            widget.destroy()
        confirmation_window.destroy()
        save_history()

    yes_button = tk.Button(button_frame, text="🩷oui🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=clear_history)
    yes_button.pack(side=tk.LEFT, padx=10)

    no_button = tk.Button(button_frame, text="🩷non🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=confirmation_window.destroy)
    no_button.pack(side=tk.RIGHT, padx=10)

# Fonction pour ouvrir la fenêtre "Analyse des sosies possibles"
def open_analysis_window(distances, best_distance):
    analysis_window = tk.Toplevel()
    analysis_window.title("🩷Analyse des sosies possibles🩷")

    # Centrer la fenêtre "Analyse des sosies possibles"
    screen_width = analysis_window.winfo_screenwidth()
    screen_height = analysis_window.winfo_screenheight()
    window_width = 670
    window_height = 400
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    analysis_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    analysis_window.config(bg=bg_color)

    # Créer un Canvas pour contenir les widgets
    canvas = tk.Canvas(analysis_window, bg=bg_color)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Ajouter une barre de défilement verticale au Canvas
    scrollbar = tk.Scrollbar(analysis_window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configurer le Canvas pour qu'il puisse faire défiler les widgets
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Créer un cadre pour l'analyse
    analysis_frame = tk.Frame(canvas, bg=bg_color)
    canvas.create_window((0, 0), window=analysis_frame, anchor='nw')

    # Ajouter un label pour l'analyse
    analysis_label = tk.Label(analysis_frame, text="🩷Sosies possibles et vos pourcentages de ressemblance🩷", font=("Times New Roman", 20), fg=text_color, bg=bg_color)
    analysis_label.pack(pady=20)

    # Ajouter les sosies possibles avec leurs pourcentages de ressemblance
    max_distance = max(distance for _, distance in distances)
    for file_name, distance in distances:
        similarity_percentage = (1 - distance / max_distance) * 100
        entry_label = tk.Label(analysis_frame, text=f"    {file_name.split('.')[0]} : {similarity_percentage:.2f}% de ressemblance", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
        entry_label.pack(anchor='w')

# Fonction pour générer le PDF
def generate_pdf(user_image_path, sosie_image_path):
    if not user_image_path or not sosie_image_path:
        messagebox.showerror("Erreur", "Veuillez charger les deux images avant de générer le PDF.")
        return

    pdf = FPDF(orientation='L', unit='mm', format='A4')  # Format paysage
    pdf.add_page()

    page_width = 297  # Largeur A4 en mm (format paysage)
    page_height = 210  # Hauteur A4 en mm (format paysage)

    try:
        # Ajouter une image de fond
        background_url = "https://i.pinimg.com/736x/3d/1a/ca/3d1aca62e84905faf872b0a36e5d3dd9.jpg"
        response = requests.get(background_url)
        background_image = Image.open(BytesIO(response.content))
        background_path = os.path.join(os.getcwd(), "background.jpg")
        background_image.save(background_path)

        pdf.image(background_path, x=0, y=0, w=page_width, h=page_height)

        # Définir la taille par défaut des images dans le PDF souvenir
        image_width_mm = 120  # largeur en mm
        image_height_mm = 130  # hauteur en mm

        # Calculer les positions pour centrer les images horizontalement et verticalement
        left_x = (page_width / 4) - (image_width_mm / 2)
        right_x = (3 * page_width / 4) - (image_width_mm / 2)

        # Ajuster la position verticale en baissant un peu les images
        left_y = (page_height / 2) - (image_height_mm / 2) + 10  # Petit décalage vertical
        right_y = (page_height / 2) - (image_height_mm / 2) + 10  # Même pour l'autre image

        # Ajouter les images avec la taille définie (elles seront maintenant beaucoup plus grandes)
        pdf.image(user_image_path, x=left_x, y=left_y, w=image_width_mm, h=image_height_mm)
        pdf.image(sosie_image_path, x=right_x, y=right_y, w=image_width_mm, h=image_height_mm)

        # Obtenir la date et l'heure actuelles
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H-%M-%S")

        # Sauvegarder le fichier PDF
        output_path = os.path.join(os.getcwd(), f"Trouve mon sosie {date_str}.pdf")
        pdf.output(output_path)

        # Vérifier si le fichier a été créé
        if os.path.exists(output_path):
            # Créer une fenêtre de notification
            notification_window = tk.Toplevel()
            notification_window.title("🩷Souvenir téléchargé avec succès🩷")

            # Centrer la fenêtre de notification
            screen_width = notification_window.winfo_screenwidth()
            screen_height = notification_window.winfo_screenheight()
            window_width = 400
            window_height = 200
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            notification_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # Définir le fond de la fenêtre en noir
            notification_window.config(bg=bg_color)

            # Ajouter un label avec le message de succès
            success_label = tk.Label(notification_window, text=f"👩‍🦰Ton souvenir a été rangé dans le répertoire courant :\n{output_path}👩🏽‍🦱", font=("Times New Roman", 14), fg=text_color, bg=bg_color, wraplength=350)
            success_label.pack(pady=20)

            # Ajouter un bouton pour accéder au fichier
            def open_file_location():
                # Ouvrir le répertoire contenant le fichier
                os.startfile(os.path.dirname(output_path))
                # Ouvrir le fichier PDF
                webbrowser.open(output_path)
                # Fermer la fenêtre de notification
                notification_window.destroy()

            access_button = tk.Button(notification_window, text="🩷Trouver mon souvenir🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=open_file_location)
            access_button.pack(pady=10)

        else:
            messagebox.showerror("Erreur", "Le fichier PDF n'a pas pu être créé.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors de la génération du PDF : {str(e)}")

# Modification de la fonction create_etape2_window
def create_etape2_window(user_name):
    global history, search_thread_active, distances

    # Créer la nouvelle fenêtre "Etape 2"
    new_window = tk.Tk()
    new_window.title("🩷Etape 2🩷")

    # Mettre la fenêtre en plein écran
    new_window.attributes("-fullscreen", True)

    # Définir le fond de la fenêtre en noir
    new_window.config(bg=bg_color)

    # Ajouter un bouton "❌" dans l'angle supérieur droit
    close_button = tk.Button(new_window, text="❌", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=lambda: open_quit_confirmation(new_window))
    close_button.place(x=new_window.winfo_screenwidth() - 10, y=10, anchor='ne')  # Positionner dans l'angle supérieur droit

    # Ajouter un bouton "📜" à gauche du bouton "❌"
    history_button = tk.Button(new_window, text="📜Historique", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=open_history_window)
    history_button.place(x=new_window.winfo_screenwidth() - 90, y=10, anchor='ne')  # Positionner à gauche du bouton "❌"

    # Créer un cadre pour les boutons "Aide" et "Accueil"
    button_frame = tk.Frame(new_window, bg=bg_color)
    button_frame.place(x=10, y=10)  # Positionner le cadre en haut à gauche avec une marge de 10px

    # Créer un bouton "🧙 Aide" dans le cadre avec taille de police 23px
    help_button = tk.Button(button_frame, text="🧙 Aide", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=open_help_video)
    help_button.pack(side=tk.LEFT, padx=5)  # Positionner le bouton à gauche avec une marge de 5px

    # Créer un bouton "🏠 Accueil" dans le cadre avec taille de police 23px
    home_button = tk.Button(button_frame, text="🏠 Accueil", font=("Times New Roman", 23), fg=text_color, bg=button_color, command=lambda: open_home_confirmation(new_window))
    home_button.pack(side=tk.LEFT, padx=5)  # Positionner le bouton à gauche avec une marge de 5px

    # Ajouter un label personnalisé au centre, légèrement vers le haut
    welcome_label = tk.Label(new_window, text=f"👩‍🦰👩🏽‍🦱Bienvenue {user_name}, charge/prends une photo pour trouver ton sosie👩‍🦰👩🏽‍🦱", font=("Times New Roman", 24), fg=text_color, bg=bg_color)
    welcome_label.pack(pady=(100, 20))  # 100px de marge en haut et 20px de marge en bas

    # Créer un label pour afficher l'image
    img_label = tk.Label(new_window, bg=bg_color)
    img_label.pack(pady=20)  # Ajouter un peu de marge sous le bouton

    # Créer un cadre pour les boutons "Charger une photo" et "Valider ma photo"
    button_frame2 = tk.Frame(new_window, bg=bg_color)
    button_frame2.pack(pady=20)  # Ajouter un peu de marge sous le bouton

    # Créer un bouton pour valider la photo
    validate_button = tk.Button(button_frame2, text="🩷Valider ma photo🩷", font=("Times New Roman", 20), fg=text_color, bg=button_color)

    # Chemin de l'image utilisateur
    user_image_path = None

    # Fonction pour charger une image depuis le PC
    def load_image():
        nonlocal user_image_path
        # Ouvrir la boîte de dialogue pour sélectionner une image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            # Charger l'image sélectionnée
            image = Image.open(file_path)

            # Redimensionner l'image pour qu'elle tienne dans la fenêtre tout en conservant les proportions
            max_size = 400
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convertir l'image pour Tkinter
            photo = ImageTk.PhotoImage(image)

            # Mettre à jour le label avec la nouvelle image
            img_label.config(image=photo)
            img_label.photo = photo  # Garder une référence de l'image pour éviter qu'elle soit détruite

            # Afficher le bouton de validation
            validate_button.pack(side=tk.RIGHT, padx=10)  # Ajouter un peu de marge à droite du bouton

            # Mettre à jour le chemin de l'image utilisateur
            user_image_path = file_path

    # Créer un bouton pour charger une image
    load_button = tk.Button(button_frame2, text="🩷Charger une photo🩷", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=load_image)
    load_button.pack(side=tk.LEFT, padx=10)  # Ajouter un peu de marge à gauche du bouton

    # Fonction pour démarrer la webcam
    def start_webcam():
        nonlocal user_image_path
        # Ouvrir la webcam
        cap = cv2.VideoCapture(0)

        # Vérifier si la webcam est ouverte
        if not cap.isOpened():
            print("Erreur : Impossible d'ouvrir la webcam")
            return

        # Créer une nouvelle fenêtre pour afficher le flux vidéo
        webcam_window = tk.Toplevel(new_window)
        webcam_window.title("🩷Webcam sayyy cheeese🩷")

        # Créer un label pour afficher le flux vidéo
        webcam_label = tk.Label(webcam_window)
        webcam_label.pack()

        # Fonction pour capturer l'image
        def capture_image():
            nonlocal user_image_path
            # Lire une image depuis la webcam
            ret, frame = cap.read()

            # Vérifier si l'image a été lue correctement
            if not ret:
                print("Erreur : Impossible de lire l'image depuis la webcam")
                return

            # Convertir l'image de BGR (OpenCV) à RGB (PIL)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir l'image en objet PIL
            image = Image.fromarray(image)

            # Redimensionner l'image pour qu'elle tienne dans la fenêtre tout en conservant les proportions
            max_size = 400
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convertir l'image pour Tkinter
            photo = ImageTk.PhotoImage(image)

            # Mettre à jour le label avec la nouvelle image
            img_label.config(image=photo)
            img_label.photo = photo  # Garder une référence de l'image pour éviter qu'elle soit détruite

            # Afficher le bouton de validation
            validate_button.pack(side=tk.RIGHT, padx=10)  # Ajouter un peu de marge à droite du bouton

            # Sauvegarder l'image temporairement pour l'analyse
            temp_image_path = "temp_image.jpg"
            image.save(temp_image_path)
            user_image_path = temp_image_path

            # Fermer la fenêtre de la webcam
            webcam_window.destroy()

        # Créer un bouton pour capturer l'image
        capture_button = tk.Button(webcam_window, text="🩷Prendre la photo🩷", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=capture_image)
        capture_button.pack(pady=10)

        # Fonction pour mettre à jour le flux vidéo
        def update_webcam():
            # Lire une image depuis la webcam
            ret, frame = cap.read()

            # Vérifier si l'image a été lue correctement
            if not ret:
                print("Erreur : Impossible de lire l'image depuis la webcam")
                return

            # Convertir l'image de BGR (OpenCV) à RGB (PIL)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convertir l'image en objet PIL
            image = Image.fromarray(image)

            # Convertir l'image pour Tkinter
            photo = ImageTk.PhotoImage(image)

            # Mettre à jour le label avec la nouvelle image
            webcam_label.config(image=photo)
            webcam_label.photo = photo  # Garder une référence de l'image pour éviter qu'elle soit détruite

            # Appeler la fonction de mise à jour après 10 ms
            webcam_window.after(10, update_webcam)

        # Démarrer la mise à jour du flux vidéo
        update_webcam()

    # Créer un bouton pour démarrer la webcam
    start_webcam_button = tk.Button(button_frame2, text="🩷Prendre une photo🩷", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=start_webcam)
    start_webcam_button.pack(side=tk.LEFT, padx=10)  # Ajouter un peu de marge à gauche du bouton

    # Fonction pour valider la photo
    def validate_photo():
        nonlocal user_image_path
        if user_image_path:
            # Masquer les boutons de chargement et de validation
            load_button.pack_forget()
            validate_button.pack_forget()
            start_webcam_button.pack_forget()

            # Déplacer l'image initialement chargée vers la gauche
            img_label.place(x=50, y=200)

            # Ajouter le nom de l'image chargée au-dessus de l'image
            img_name_label = tk.Label(new_window, text=os.path.basename(user_image_path), font=("Times New Roman", 20), fg=text_color, bg=bg_color)
            img_name_label.place(x=50, y=160)
            # Ajouter un texte "Chargement..."
            global progress_label
            progress_label = tk.Label(new_window, text="🩷Chargement... 0%🩷", font=("Times New Roman", 20), fg=text_color, bg=bg_color)
            progress_label.place(x=new_window.winfo_screenwidth() // 2 + 100, y=250)

            # Ajouter une barre de progression (Progressbar) sous le texte "Chargement..."
            global progress_var
            progress_var = tk.DoubleVar()
            progress = ttk.Progressbar(new_window, orient="horizontal", length=400, mode="determinate", variable=progress_var)
            progress.place(x=new_window.winfo_screenwidth() // 2 + 100, y=300)
            progress.start()  # Démarrer l'animation de la barre

            # Ajouter un bouton "Annuler"
            cancel_button = tk.Button(new_window, text="🩷Annuler🩷", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=open_cancel_confirmation)
            cancel_button.place(x=new_window.winfo_screenwidth() // 2 + 100, y=350)

            # Lancer la recherche de sosie dans un thread séparé
            def search_sosie():
                global search_thread_active, best_distance
                search_thread_active = True
                best_match, distances, best_distance, max_distance = find_sosie(user_image_path, new_window)
                if search_thread_active and best_match:
                    # Arrêter la barre de progression
                    progress.stop()
                    # Mettre à jour le texte de la barre de progression
                    progress_label.config(text="🩷Chargement terminé🩷")

                    # Charger l'image du sosie
                    sosie_image_path = os.path.join(faces_folder, best_match)
                    sosie_image = Image.open(sosie_image_path)

                    # Redimensionner l'image pour qu'elle tienne dans la fenêtre tout en conservant les proportions
                    max_size = 400
                    sosie_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    # Convertir l'image pour Tkinter
                    sosie_photo = ImageTk.PhotoImage(sosie_image)

                    # Créer un label pour afficher l'image du sosie
                    sosie_label = tk.Label(new_window, image=sosie_photo, bg=bg_color)
                    sosie_label.photo = sosie_photo  # Garder une référence de l'image pour éviter qu'elle soit détruite
                    sosie_label.place(x=new_window.winfo_screenwidth() // 2 + 100, y=200)

                    # Ajouter le nom du sosie au-dessus de l'image
                    sosie_name_label = tk.Label(new_window, text=best_match.split('.')[0], font=("Times New Roman", 20), fg=text_color, bg=bg_color)
                    sosie_name_label.place(x=new_window.winfo_screenwidth() // 2 + 100, y=165)

                    # Créer un cadre pour contenir les boutons "search", "restart" et "Analyse"
                    button_frame3 = tk.Frame(new_window, bg=bg_color)
                    button_frame3.pack(side=tk.BOTTOM, pady=10)  # Placer le cadre en bas de la page avec un peu de marge

                    # Ajouter le bouton pour rechercher le sosie sur Google
                    search_button = tk.Button(button_frame3, text="🔍Recherche", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=lambda: webbrowser.open(f"https://www.google.com/search?q={best_match.split('.')[0]}"))
                    search_button.pack(side=tk.LEFT, padx=10)  # Placer à gauche du bouton "restart"

                    # Ajouter un bouton "Recommencer" pour réinitialiser la fenêtre
                    restart_button = tk.Button(button_frame3, text="🔄Recommencer", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=lambda: open_restart_confirmation(new_window, user_name))
                    restart_button.pack(side=tk.LEFT, padx=10)  # Placer à côté du bouton "search"

                    # Ajouter un bouton "Analyse"
                    analyse_button = tk.Button(button_frame3, text="📜Analyse", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=lambda: open_analysis_window(distances, best_distance))
                    analyse_button.pack(side=tk.LEFT, padx=10)  # Placer à côté du bouton "restart"

                    # Ajouter le bouton "Souvenir"
                    surprise_button = tk.Button(button_frame3, text="🔮Souvenir", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=lambda: generate_pdf(user_image_path, sosie_image_path))
                    surprise_button.pack(side=tk.LEFT, padx=10)  # Placer à côté du bouton "Analyse"

                    # Masquer la barre de progression
                    progress.place_forget()
                    progress_label.place_forget()
                    cancel_button.place_forget()

                    # Ajouter l'entrée à l'historique
                    file_name = best_match
                    similarity_percentage = (1 - best_distance / max_distance) * 100
                    history.append(f"    {os.path.basename(user_image_path)} 💖 {best_match.split('.')[0]} 💖 {similarity_percentage:.2f}% de ressemblance")
                    save_history()

                elif search_thread_active:
                    # Arrêter la barre de progression
                    progress.stop()
                    # Mettre à jour le texte de la barre de progression
                    progress_label.config(text="🩷Chargement terminé🩷")

                    # Afficher un message si aucun sosie n'est trouvé
                    no_sosie_label = tk.Label(new_window, text="🩷Aucun sosie trouvé, changez de photo🩷", font=("Times New Roman", 20), fg=text_color, bg=bg_color)
                    no_sosie_label.place(x=new_window.winfo_screenwidth() // 2 + 100, y=200)

                    # Masquer la barre de progression
                    progress.place_forget()
                    progress_label.place_forget()
                    cancel_button.place_forget()

            # Lancer la recherche de sosie dans un thread séparé
            global search_thread
            search_thread = threading.Thread(target=search_sosie)
            search_thread.start()

            # Assurer que la recherche s'arrête si la fenêtre est fermée
            new_window.protocol("WM_DELETE_WINDOW", lambda: on_window_close(new_window))

    # Fonction pour ouvrir la fenêtre de confirmation d'annulation
    def open_cancel_confirmation():
        # Créer une nouvelle fenêtre pour la confirmation
        confirmation_window = tk.Toplevel(new_window)
        confirmation_window.title("🩷Annuler la recherche?🩷")

        # Centrer la fenêtre de confirmation
        screen_width = confirmation_window.winfo_screenwidth()
        screen_height = confirmation_window.winfo_screenheight()
        window_width = 400
        window_height = 200
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        confirmation_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Définir le fond de la fenêtre en noir
        confirmation_window.config(bg=bg_color)

        # Ajouter un label avec la question
        label = tk.Label(confirmation_window, text="🩷Souhaites-tu annuler la recherche?🩷", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
        label.pack(pady=20)

        # Créer un cadre pour les boutons
        button_frame = tk.Frame(confirmation_window, bg=bg_color)
        button_frame.pack(pady=10)

        # Créer les boutons "🩷oui🩷" et "🩷non🩷"
        def cancel_search():
            global search_thread_active
            search_thread_active = False
            search_thread.join(timeout=0.1)  # Attendre que le thread se termine
            confirmation_window.destroy()  # Fermer la fenêtre de confirmation
            restart_experience(new_window, user_name)  # Revenir à l'état initial

        yes_button = tk.Button(button_frame, text="🩷oui🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=cancel_search)
        yes_button.pack(side=tk.LEFT, padx=10)

        no_button = tk.Button(button_frame, text="🩷non🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=confirmation_window.destroy)
        no_button.pack(side=tk.RIGHT, padx=10)

    # Fonction pour gérer la fermeture de la fenêtre
    def on_window_close(new_window):
        global search_thread_active
        search_thread_active = False
        if search_thread.is_alive():
            search_thread.join(timeout=0.1)  # Attendre que le thread se termine
        new_window.destroy()  # Fermer la fenêtre

    # Associer la fonction de validation au bouton
    validate_button.config(command=validate_photo)

    # Lancer la boucle principale de la nouvelle fenêtre
    new_window.mainloop()

# Fonction pour redémarrer l'expérience
def restart_experience(new_window, user_name):
    global distances
    distances = []  # Vider la liste des distances à chaque fois que l'on utilise l'option "Recommencer"
    new_window.destroy()  # Fermer la fenêtre "Etape 2"
    create_etape2_window(user_name)  # Recréer la fenêtre "Etape 2" pour recommencer

# Fonction pour ouvrir la fenêtre de confirmation de recommencement
def open_restart_confirmation(parent_window, user_name):
    # Créer une nouvelle fenêtre pour la confirmation
    confirmation_window = tk.Toplevel(parent_window)
    confirmation_window.title("🩷Recommencer?🩷")

    # Centrer la fenêtre de confirmation
    screen_width = confirmation_window.winfo_screenwidth()
    screen_height = confirmation_window.winfo_screenheight()
    window_width = 400
    window_height = 200
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    confirmation_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    confirmation_window.config(bg=bg_color)

    # Ajouter un label avec la question
    label = tk.Label(confirmation_window, text="Souhaites-tu recommencer?", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
    label.pack(pady=20)

    # Créer un cadre pour les boutons
    button_frame = tk.Frame(confirmation_window, bg=bg_color)
    button_frame.pack(pady=10)

    # Créer les boutons "🩷oui🩷" et "🩷non🩷"
    yes_button = tk.Button(button_frame, text="🩷oui🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=lambda: restart_experience(parent_window, user_name))
    yes_button.pack(side=tk.LEFT, padx=10)

    no_button = tk.Button(button_frame, text="🩷non🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=confirmation_window.destroy)
    no_button.pack(side=tk.RIGHT, padx=10)

# Fonction pour ouvrir la fenêtre de confirmation de retour à l'accueil
def open_home_confirmation(parent_window):
    # Créer une nouvelle fenêtre pour la confirmation
    confirmation_window = tk.Toplevel(parent_window)
    confirmation_window.title("🩷Accueil?🩷")

    # Centrer la fenêtre de confirmation
    screen_width = confirmation_window.winfo_screenwidth()
    screen_height = confirmation_window.winfo_screenheight()
    window_width = 400
    window_height = 200
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    confirmation_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    confirmation_window.config(bg=bg_color)

    # Ajouter un label avec la question
    label = tk.Label(confirmation_window, text="🩷Souhaites-tu revenir à la page précédente?🩷", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
    label.pack(pady=20)

    # Créer un cadre pour les boutons
    button_frame = tk.Frame(confirmation_window, bg=bg_color)
    button_frame.pack(pady=10)

    # Créer les boutons "🩷oui🩷" et "🩷non🩷"
    yes_button = tk.Button(button_frame, text="🩷oui🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=lambda: go_back_to_main(parent_window, confirmation_window))
    yes_button.pack(side=tk.LEFT, padx=10)

    no_button = tk.Button(button_frame, text="🩷non🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=confirmation_window.destroy)
    no_button.pack(side=tk.RIGHT, padx=10)

# Fonction pour revenir à la page principale
def go_back_to_main(parent_window, confirmation_window):
    global search_thread_active
    search_thread_active = False
    if 'search_thread' in globals() and search_thread.is_alive():
        search_thread.join(timeout=0.1)
    confirmation_window.destroy()
    parent_window.destroy()
    create_window()

# Fonction pour afficher la fenêtre de démarrage
def show_start_window():
    # Créer la fenêtre de démarrage
    start_window = tk.Tk()
    start_window.title("🩷Bienvenue dans notre app de reconnaissance faciale -Camilia et Sarah🩷")

    # Centrer la fenêtre de démarrage
    screen_width = start_window.winfo_screenwidth()
    screen_height = start_window.winfo_screenheight()
    window_width = 630
    window_height = 470
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    start_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    start_window.config(bg=bg_color)

    # Charger l'image depuis l'URL
    image_url = "https://i.pinimg.com/736x/0e/32/c5/0e32c5f7ea60766481490ceff20961d7.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Redimensionner l'image pour qu'elle tienne dans la fenêtre tout en conservant les proportions
    max_size = 600
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convertir l'image pour Tkinter
    photo = ImageTk.PhotoImage(image)

    # Créer un label pour afficher l'image
    image_label = tk.Label(start_window, image=photo, bg=bg_color)
    image_label.pack(pady=20)

    # Créer un bouton "⭐START⭐"
    start_button = tk.Button(start_window, text="⭐START⭐", font=("Times New Roman", 20), fg=text_color, bg=button_color, command=lambda: open_main_window(start_window))
    start_button.pack(pady=20)

    # Lancer la boucle principale de la fenêtre de démarrage
    start_window.mainloop()

# Fonction pour ouvrir la fenêtre principale
def open_main_window(start_window):
    start_window.destroy()  # Fermer la fenêtre de démarrage
    create_window()  # Ouvrir la fenêtre principale

# Fonction pour ouvrir la fenêtre de personnalisation
def open_customization_window():
    global new_window
    # Créer la fenêtre de personnalisation
    customization_window = tk.Toplevel()
    customization_window.title("🩷Personnalisation🩷")

    # Centrer la fenêtre de personnalisation
    screen_width = customization_window.winfo_screenwidth()
    screen_height = customization_window.winfo_screenheight()
    window_width = 400
    window_height = 400
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    customization_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Définir le fond de la fenêtre en noir
    customization_window.config(bg=bg_color)

    # Ajouter un label pour le titre
    title_label = tk.Label(customization_window, text="Personnalisation", font=("Times New Roman", 20), fg=text_color, bg=bg_color)
    title_label.pack(pady=10)

    # Ajouter un bouton pour choisir la couleur du fond
    def choose_bg_color():
        global bg_color
        bg_color = colorchooser.askcolor(title="Choisir la couleur du fond")[1]
        apply_colors()

    bg_color_button = tk.Button(customization_window, text="🩷Fonds🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=choose_bg_color)
    bg_color_button.pack(pady=10)

    # Ajouter un bouton pour choisir la couleur des boutons
    def choose_button_color():
        global button_color
        button_color = colorchooser.askcolor(title="Choisir la couleur des boutons")[1]
        apply_colors()

    button_color_button = tk.Button(customization_window, text="🩷Boutons🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=choose_button_color)
    button_color_button.pack(pady=10)

    # Ajouter un bouton pour choisir la couleur des écritures
    def choose_text_color():
        global text_color
        text_color = colorchooser.askcolor(title="Choisir la couleur des écritures")[1]
        apply_colors()

    text_color_button = tk.Button(customization_window, text="🩷Ecritures🩷", font=("Times New Roman", 16), fg=text_color, bg=button_color, command=choose_text_color)
    text_color_button.pack(pady=10)

    # Ajouter un menu déroulant pour sélectionner le thème
    def change_theme(theme):
        global bg_color, button_color, text_color, entry_frame_color, entry_bg_color
        if theme == "Par défaut":
            bg_color = "black"
            button_color = "black"
            text_color = "#4cbae7"
            entry_frame_color = "#ff82e6"
            entry_bg_color = "black"
        elif theme == "🩷Clair🩷":
            bg_color = "white"
            button_color = "white"
            text_color = "black"
            entry_frame_color = "lightgrey"
            entry_bg_color = "white"
        elif theme == "🩷Sombre🩷":
            bg_color = "black"
            button_color = "black"
            text_color = "white"
            entry_frame_color = "darkgrey"
            entry_bg_color = "white"
        elif theme == "🩷Camilia🩷":
            bg_color = "#d7a8f0"
            button_color = "#65208a"
            text_color = "white"
            entry_frame_color = "black"
            entry_bg_color = "black"
        elif theme == "🩷Sarah🩷":
            bg_color = "#a8d7f0"
            button_color = "#52a1cc"
            text_color = "#660856"
            entry_frame_color = "black"
            entry_bg_color = "black"
        apply_colors()

    theme_label = tk.Label(customization_window, text="Thème", font=("Times New Roman", 16), fg=text_color, bg=bg_color)
    theme_label.pack(pady=10)

    theme_var = tk.StringVar(value="Par défaut")
    theme_menu = tk.OptionMenu(customization_window, theme_var, "Par défaut", "🩷Clair🩷", "🩷Sombre🩷", "🩷Camilia🩷", "🩷Sarah🩷", command=change_theme)
    theme_menu.config(font=("Times New Roman", 16), fg=text_color, bg=button_color)
    theme_menu.pack(pady=10)

    # Fonction pour appliquer les couleurs sélectionnées
    def apply_colors():
        # Appliquer les couleurs à toutes les fenêtres
        root.config(bg=bg_color)
        for widget in root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(fg=text_color, bg=button_color)
            elif isinstance(widget, tk.Label):
                widget.config(fg=text_color, bg=bg_color)
            elif isinstance(widget, tk.Frame):
                widget.config(bg=bg_color)
            elif isinstance(widget, tk.Entry):
                widget.config(fg=text_color, bg=entry_bg_color)
            elif isinstance(widget, tk.Toplevel):
                widget.config(bg=bg_color)
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button):
                        child.config(fg=text_color, bg=button_color)
                    elif isinstance(child, tk.Label):
                        child.config(fg=text_color, bg=bg_color)
                    elif isinstance(child, tk.Frame):
                        child.config(bg=bg_color)
                    elif isinstance(child, tk.Entry):
                        child.config(fg=text_color, bg=entry_bg_color)

        # Appliquer les couleurs à la fenêtre de personnalisation
        customization_window.config(bg=bg_color)
        for widget in customization_window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(fg=text_color, bg=button_color)
            elif isinstance(widget, tk.Label):
                widget.config(fg=text_color, bg=bg_color)
            elif isinstance(widget, tk.Frame):
                widget.config(bg=bg_color)
            elif isinstance(widget, tk.OptionMenu):
                widget.config(fg=text_color, bg=button_color)

        # Appliquer les couleurs à la fenêtre "Etape 2"
        if 'new_window' in globals():
            new_window.config(bg=bg_color)
            for widget in new_window.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.config(fg=text_color, bg=button_color)
                elif isinstance(widget, tk.Label):
                    widget.config(fg=text_color, bg=bg_color)
                elif isinstance(widget, tk.Frame):
                    widget.config(bg=bg_color)
                elif isinstance(widget, tk.Entry):
                    widget.config(fg=text_color, bg=entry_bg_color)

        # Sauvegarder les préférences de personnalisation
        save_preferences()

    # Capturer toutes les entrées de l'utilisateur pour garder la fenêtre en premier plan
    customization_window.grab_set()

    # Lancer la boucle principale de la fenêtre de personnalisation
    customization_window.mainloop()

# Fonction pour sauvegarder l'historique
def save_history():
    with open('history.json', 'w') as f:
        json.dump(history, f)

# Fonction pour charger l'historique
def load_history():
    global history
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            history = json.load(f)

# Fonction pour sauvegarder les préférences de personnalisation
def save_preferences():
    preferences = {
        'bg_color': bg_color,
        'button_color': button_color,
        'text_color': text_color,
        'entry_frame_color': entry_frame_color,
        'entry_bg_color': entry_bg_color
    }
    with open('preferences.json', 'w') as f:
        json.dump(preferences, f)

# Fonction pour charger les préférences de personnalisation
def load_preferences():
    global bg_color, button_color, text_color, entry_frame_color, entry_bg_color
    if os.path.exists('preferences.json'):
        with open('preferences.json', 'r') as f:
            preferences = json.load(f)
            bg_color = preferences.get('bg_color', bg_color)
            button_color = preferences.get('button_color', button_color)
            text_color = preferences.get('text_color', text_color)
            entry_frame_color = preferences.get('entry_frame_color', entry_frame_color)
            entry_bg_color = preferences.get('entry_bg_color', entry_bg_color)

# Lancer la fonction pour créer la fenêtre initiale
if __name__ == "__main__":
    history = []  # Initialiser l'historique globalement
    search_thread_active = True  # Initialiser le drapeau pour contrôler l'état du thread de recherche
    bg_color = "black"
    button_color = "black"  # Changer cette ligne pour mettre la couleur des boutons en noir
    text_color = "#4cbae7"
    entry_frame_color = "#ff82e6"
    entry_bg_color = "black"
    load_history()  # Charger l'historique
    load_preferences()  # Charger les préférences de personnalisation
    show_start_window()  # Afficher la fenêtre de démarrage
