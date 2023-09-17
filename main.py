import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import io
from PIL import Image, ImageTk

import problem1

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 6))

canvas_elem = sg.Image(key='-CANVAS-')
# Define the layout
left_column = [
    [sg.Text("Select an option:")],
    [sg.Combo(["Fibonacci", "Razão áurea", "Descida máxima", "Descida de gradiente", "Newton", "Quasi-Newton", "Brent"], key='-OPTION-', enable_events=True)],
    [sg.Text("Select a number:")],
    [sg.Combo(["3", "5", "Todas"], key='-NUMBER-', enable_events=True)],
    [sg.Text("K inicial:")],
    [sg.InputText(key='K', enable_events=True, justification='right', size=(10, 1))],
    [sg.Text("Tau inicial:")],
    [sg.InputText(key='tau', enable_events=True, justification='right', size=(10, 1))],
    [sg.Text("Alpha:")],
    [sg.InputText(key='alpha', enable_events=True, justification='right', size=(10, 1))],
    [sg.Text("Iterações:")],
    [sg.InputText(key='iters', enable_events=True, justification='right', size=(10, 1))],
    [sg.Button("Calculate")],
    [sg.Text("", key='-RESULT-')],
]

# Event loop
layout = [
    [
        sg.Column(left_column, element_justification='center'),
        sg.VerticalSeparator(),
        sg.Column([[canvas_elem]], element_justification='center', size=(800, 600))
    ]
]
# Create the window
window = sg.Window("Dropdown GUI", layout, resizable=True)

selected_option = None
selected_number = None

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Calculate":
        selected_option = values['-OPTION-']
        selected_number = values['-NUMBER-']
        k0 = None
        tau0 = None
        iters = None
        alpha = None
        try:
            k0 = float(values['K'])
            tau0 = float(values['tau'])
            iters = int(values['iters'])
            alpha = float(values['alpha'])
        except Exception as e:
            print(e)
            result_text = f"Valor inválido para Iterações: {iters}, Alpha: {alpha} ou Tau: {tau0} ou K: {k0}\n{e}"
            window['-RESULT-'].update(result_text)
            continue
        
        # Update the Matplotlib plot
        errors, k, tau = problem1.gradient_descent(alpha, iters, k0, tau0)
        # Downsample the errors array if it's too large

        result_text = f"Tau ótimo: {tau:.2f}, K ótimo: {k:.2f}, Erro: {errors[-1]}"
        window['-RESULT-'].update(result_text)
        ax.clear()
        ax.plot(errors)
        ax.set_title("Matplotlib Plot")
        ax.set_xlabel("Iteração")
        ax.set_ylabel("Erro")
        ax.set_title(f"Valores ótimos: Tau: {tau:.2f}, K: {k:.2f}; Erro: {errors[-1]}")

        result_text = f"Tau ótimo: {tau:.2f}, K ótimo: {k:.2f}, Erro: {errors[-1]}"
        window['-RESULT-'].update(result_text)

        # Convert Matplotlib figure to an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img.thumbnail((800, 600))  # Resize the image to fit the canvas
        img_data = ImageTk.PhotoImage(img)

        # Update the canvas with the Matplotlib plot image
        canvas_elem.update(data=img_data)

# Close the window
window.close()

