import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
# Real-time spectrum analyzer
#ChatGPTo1-mini and Tom Moir 11/11/2024
# Create the main window
root = tk.Tk()
root.title("Real-time Signal and Power Spectrum")

# Get the list of input devices
devices = sd.query_devices()
input_devices = [d for d in devices if d['max_input_channels'] > 0]
device_names = [f"{d['index']}: {d['name']}" for d in input_devices]

# Variables
is_recording = False
stream = None
data_queue = queue.Queue()

# Control frame
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

# Device selection
device_label = tk.Label(control_frame, text="Input Device:")
device_label.pack(side=tk.LEFT)

device_var = tk.StringVar()
device_combo = ttk.Combobox(control_frame, textvariable=device_var, state="readonly")
device_combo['values'] = device_names
device_combo.current(0)
device_combo.pack(side=tk.LEFT)

# Buffer length
buffer_label = tk.Label(control_frame, text="Buffer Length:")
buffer_label.pack(side=tk.LEFT)

buffer_var = tk.IntVar(value=1024)
buffer_spin = tk.Spinbox(control_frame, from_=256, to=8192, increment=256, textvariable=buffer_var)
buffer_spin.pack(side=tk.LEFT)

# FFT length
fft_label = tk.Label(control_frame, text="FFT Length:")
fft_label.pack(side=tk.LEFT)

fft_var = tk.IntVar(value=1024)
fft_spin = tk.Spinbox(control_frame, from_=256, to=8192, increment=256, textvariable=fft_var)
fft_spin.pack(side=tk.LEFT)

# Sampling rate selection
sampling_rates = [8000, 16000, 22050, 32000, 44100, 48000, 96000]
sampling_rate_label = tk.Label(control_frame, text="Sampling Rate:")
sampling_rate_label.pack(side=tk.LEFT)

sampling_rate_var = tk.IntVar(value=44100)
sampling_rate_combo = ttk.Combobox(
    control_frame, textvariable=sampling_rate_var, values=sampling_rates, state="readonly"
)
sampling_rate_combo.current(sampling_rates.index(44100))
sampling_rate_combo.pack(side=tk.LEFT)

# Averaging options
average_frame = tk.Frame(root)
average_frame.pack(side=tk.TOP, fill=tk.X)

average_var = tk.BooleanVar()
average_check = tk.Checkbutton(average_frame, text="Average Spectrum", variable=average_var)
average_check.pack(side=tk.LEFT)

averaging_type_label = tk.Label(average_frame, text="Averaging Type:")
averaging_type_label.pack(side=tk.LEFT)

averaging_type_var = tk.StringVar(value="Linear")
averaging_type_combo = ttk.Combobox(
    average_frame, textvariable=averaging_type_var, values=["Linear", "Exponential"], state="readonly"
)
averaging_type_combo.pack(side=tk.LEFT)

# Start and Stop buttons
start_button = tk.Button(control_frame, text="Start")
start_button.pack(side=tk.LEFT)

stop_button = tk.Button(control_frame, text="Stop")
stop_button.pack(side=tk.LEFT)

# Matplotlib figures
fig1 = plt.Figure(figsize=(6, 2))
ax1 = fig1.add_subplot(111)
line1, = ax1.plot([], [])
ax1.set_title("Time-Domain Signal")
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, buffer_var.get())
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Amplitude")

fig2 = plt.Figure(figsize=(6, 2))
ax2 = fig2.add_subplot(111)
line2, = ax2.plot([], [])
ax2.set_title("Power Spectrum")
ax2.set_ylim(-100, 0)
ax2.set_xlim(0, 8000)  # This will be updated based on sampling rate
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude (dB)")

# Canvas for plots
canvas_frame1 = tk.Frame(root)
canvas_frame1.pack(side=tk.TOP)
canvas1 = FigureCanvasTkAgg(fig1, master=canvas_frame1)
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

canvas_frame2 = tk.Frame(root)
canvas_frame2.pack(side=tk.TOP)
canvas2 = FigureCanvasTkAgg(fig2, master=canvas_frame2)
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Averaging variables
averaged_spectrum = None
average_count = 0
alpha = 0.1  # Exponential averaging factor

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_data = indata[:, 0]
    data_queue.put(audio_data.copy())

def update_plots():
    global averaged_spectrum, average_count
    if not data_queue.empty():
        audio_data = data_queue.get()

        # Update time-domain plot
        line1.set_data(np.arange(len(audio_data)), audio_data)
        ax1.set_xlim(0, len(audio_data))
        ax1.set_ylim(np.min(audio_data), np.max(audio_data))

        # Compute FFT
        fft_length = fft_var.get()
        sampling_rate = sampling_rate_var.get()

        # Zero-padding or truncating audio data to match FFT length
        if len(audio_data) < fft_length:
            audio_data_padded = np.zeros(fft_length)
            audio_data_padded[:len(audio_data)] = audio_data
        else:
            audio_data_padded = audio_data[:fft_length]

        fft_data = np.fft.fft(audio_data_padded)
        fft_freq = np.fft.fftfreq(fft_length, d=1.0 / sampling_rate)
        fft_magnitude = np.abs(fft_data)

        # Consider only the positive frequencies
        pos_mask = fft_freq >= 0
        fft_freq = fft_freq[pos_mask]
        fft_magnitude = fft_magnitude[pos_mask]

        # Averaging
        if average_var.get():
            averaging_type = averaging_type_var.get()
            if averaged_spectrum is None:
                averaged_spectrum = fft_magnitude.copy()
                average_count = 1
            else:
                if averaging_type == "Linear":
                    average_count += 1
                    averaged_spectrum = averaged_spectrum + (fft_magnitude - averaged_spectrum) / average_count
                elif averaging_type == "Exponential":
                    averaged_spectrum = alpha * fft_magnitude + (1 - alpha) * averaged_spectrum
            fft_magnitude_to_plot = averaged_spectrum
        else:
            averaged_spectrum = None
            average_count = 0
            fft_magnitude_to_plot = fft_magnitude

        # Convert to dB
        epsilon = 1e-10
        fft_magnitude_db = 20 * np.log10(fft_magnitude_to_plot + epsilon)

        # Update power spectrum plot
        line2.set_data(fft_freq, fft_magnitude_db)
        ax2.set_xlim(0, sampling_rate / 2)
        ax2.set_ylim(np.min(fft_magnitude_db), np.max(fft_magnitude_db))

        # Redraw plots
        canvas1.draw_idle()
        canvas2.draw_idle()
    # Schedule the function to be called again
    if is_recording:
        root.after(10, update_plots)

def start_recording():
    global is_recording, stream, averaged_spectrum, average_count
    if not is_recording:
        is_recording = True
        device_selection = device_var.get()
        device_index = int(device_selection.split(':')[0])
        buffer_length = buffer_var.get()
        fft_length = fft_var.get()
        samplerate = sampling_rate_var.get()

        # Update frequency axis limits
        ax2.set_xlim(0, samplerate / 2)

        # Reset averaging variables
        averaged_spectrum = None
        average_count = 0

        # Start the stream
        try:
            stream = sd.InputStream(
                device=device_index,
                channels=1,
                callback=audio_callback,
                blocksize=buffer_length,
                samplerate=samplerate,
                dtype='float32'
            )
            stream.start()
            update_plots()
        except Exception as e:
            print("Error starting stream:", e)
            is_recording = False

def stop_recording():
    global is_recording, stream
    if is_recording:
        is_recording = False
        if stream is not None:
            stream.stop()
            stream.close()
            stream = None

start_button.config(command=start_recording)
stop_button.config(command=stop_recording)

# Run the main loop
root.mainloop()
