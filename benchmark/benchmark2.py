import time
import psutil
import platform
import socket
import subprocess
import pandas as pd
import cpuinfo  
from ollama import Client
import concurrent.futures

# File di log
LOG_FILENAME = "benchmark.txt"

# Funzione per stampare e salvare log
def log_print(message):
    print(message)
    with open(LOG_FILENAME, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

# Conversazioni: più domande per ogni mini-dialogo
CONVERSATIONS = [
    [
        "Ciao! Sai contare da uno a dieci?",
        "Bravissimo! E qual è la capitale d'Italia?",
        "Puoi spiegarmi cos'è una stella?"
    ],
    [
        "Ciao! Cos'è il Sole?",
        "Perché la luna ci segue?",
        "Quanti pianeti ci sono?"
    ],
    [
        "Ciao! Perché il cielo è blu?",
        "Come nascono i pesci?",
        "Perché le foglie cadono?"
    ]
]

EXCEL_FILENAME = "benchmark_results.xlsx"
SUMMARY_FILENAME = "benchmark_summary.xlsx"
TIMEOUT_SEC = 30  # Timeout per risposta modello

client = Client()

def format_gb(value):
    return value / (1024**3)

def get_gpu_info():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        )
        gpu_name = output.decode().strip()
        return gpu_name if gpu_name else "GPU NVIDIA non rilevata"
    except Exception:
        try:
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return "Apple Silicon GPU (Metal)"
        except:
            pass
        return "Nessuna GPU rilevata"

def extract_cpu_type(cpu_brand):
    if "Intel" in cpu_brand:
        for part in cpu_brand.split():
            if part.startswith(("i3", "i5", "i7", "i9")):
                return part
    elif "Ryzen" in cpu_brand:
        parts = cpu_brand.split()
        for i, part in enumerate(parts):
            if "Ryzen" in part and i + 1 < len(parts):
                return part + " " + parts[i + 1]
    return "N/D"

def get_system_info():
    cpu_freq = psutil.cpu_freq()
    cpu_info = cpuinfo.get_cpu_info()
    cpu_brand = cpu_info.get("brand_raw", "N/D")
    cpu_type = extract_cpu_type(cpu_brand)

    return {
        "host": socket.gethostname(),
        "cpu_brand": cpu_brand,
        "cpu_type": cpu_type,
        "cpu_freq_base_mhz": round(cpu_freq.min, 2) if cpu_freq else "N/D",
        "cpu_freq_max_mhz": round(cpu_freq.max, 2) if cpu_freq else "N/D",
        "cpu_freq_current_mhz": round(cpu_freq.current, 2) if cpu_freq else "N/D",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(format_gb(psutil.virtual_memory().total), 2),
        "os": platform.system() + " " + platform.release(),
        "architecture": platform.machine(),
        "gpu": get_gpu_info()
    }

def get_installed_models():
    try:
        models_data = client.list()
        return [m.get("model") or m.get("name") for m in models_data.get("models", []) if m.get("model") or m.get("name")]
    except Exception as e:
        log_print(f"❌ Errore nel recupero dei modelli: {e}")
        return []

def ask_single_question(model_name, question, history, timeout_sec=TIMEOUT_SEC):
    log_print(f"➡️ Domanda: {question}")

    messages = [{"role": "system", "content": "Parla con un bambino di 6-10 anni. Sii gentile, semplice e chiaro."}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        cpu_before = psutil.cpu_percent(interval=1)
        ram_before = psutil.virtual_memory().used
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(client.chat, model=model_name, messages=messages)
            response = future.result(timeout=timeout_sec)

        end_time = time.time()
        cpu_after = psutil.cpu_percent(interval=1)
        ram_after = psutil.virtual_memory().used

        result = {
            "model": model_name,
            "question": question,
            "time_sec": round(end_time - start_time, 2),
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "ram_before": round(format_gb(ram_before), 2),
            "ram_after": round(format_gb(ram_after), 2),
            "response": response["message"]["content"]
        }

        log_print(f"✅ {model_name} | Tempo: {result['time_sec']} sec")
        log_print(f"💬 Risposta: {response['message']['content'][:150]}...\n")
        return result, {"role": "assistant", "content": response["message"]["content"]}

    except concurrent.futures.TimeoutError:
        log_print(f"⏱️ Timeout su: {question}")
        return {
            "model": model_name,
            "question": question,
            "error": f"Timeout dopo {timeout_sec} secondi"
        }, None
    except Exception as e:
        log_print(f"❌ Errore: {model_name} | {question} → {str(e)}")
        return {
            "model": model_name,
            "question": question,
            "error": str(e)
        }, None

def save_to_excel(results, system_info, filename=EXCEL_FILENAME):
    df = pd.DataFrame(results)
    for key, value in system_info.items():
        df[key] = value
    df.to_excel(filename, index=False)
    log_print(f"\n📁 Risultati salvati in {filename}")

def save_summary_to_excel(results, filename=SUMMARY_FILENAME):
    summary_data = []

    for r in results:
        row = {
            "Modello": r["model"],
            "Domanda": r["question"]
        }
        if "error" in r:
            row["Tempo"] = r["error"]
            row["Risposta"] = ""
        else:
            row["Tempo"] = f"{r['time_sec']} sec"
            row["Risposta"] = r['response'][:150] + "..." if len(r['response']) > 150 else r['response']
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_excel(filename, index=False)
    log_print(f"\n📊 Statistiche dettagliate salvate in {filename}")

if __name__ == "__main__":
    with open(LOG_FILENAME, "w", encoding="utf-8") as f:
        f.write("=== BENCHMARK LOG ===\n\n")

    log_print("📊 Avvio benchmark per modelli Ollama con domande singole\n")

    system_info = get_system_info()
    log_print("🔍 INFO SISTEMA:")
    for k, v in system_info.items():
        log_print(f"   {k}: {v}")

    models = get_installed_models()
    if not models:
        log_print("❌ Nessun modello trovato. Usa 'ollama run <modello>' prima di eseguire il benchmark.")
        exit(1)

    all_results = []
    for model in models:
        log_print(f"\n🧪 Inizio test per il modello: {model}\n")
        for convo in CONVERSATIONS:
            history = []
            for question in convo:
                result, assistant_reply = ask_single_question(model, question, history)
                all_results.append(result)
                if assistant_reply:
                    history.append({"role": "user", "content": question})
                    history.append(assistant_reply)
                else:
                    continue

    log_print("\n🧾 RISULTATI COMPLETI:")
    for res in all_results:
        if "error" in res:
            log_print(f"❌ {res['model']} | Domanda: {res['question']} → ERRORE: {res['error']}")
        else:
            log_print(f"✅ {res['model']} | Domanda: {res['question']} | Tempo: {res['time_sec']} sec")

    save_to_excel(all_results, system_info)
    save_summary_to_excel(all_results)
