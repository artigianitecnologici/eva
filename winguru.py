import requests
from bs4 import BeautifulSoup

# URL della pagina
URL = "http://wap2.windguru.cz/view.php?&sc=210597&m=3&n=&from=search&start=0&full=1"

def fetch_wind_data():
    """Scarica i dati del vento da Windguru e stampa le previsioni favorevoli."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Sto recuperando i dati da: {URL}\n")
    
    try:
        response = requests.get(URL, headers=headers)

        if response.status_code != 200:
            print(f"Errore nel recuperare la pagina: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        # *** MODIFICA CHIAVE: Trova la tabella in base al contenuto ***
        # 1. Trova la cella che contiene la stringa "Wind speed (knots)"
        wind_row_header = soup.find('td', string='Wind speed (knots)')
        
        if not wind_row_header:
            print("Errore: Impossibile trovare la cella d'intestazione 'Wind speed (knots)'.")
            return

        # 2. Risali al tag 'table' genitore
        data_table = wind_row_header.find_parent('table')

        if not data_table:
            print("Errore: Impossibile trovare la tabella dei dati (tabella principale).")
            return
        
        # 3. Trova tutte le righe dei dati all'interno di QUESTA tabella
        # Le righe dei dati effettivi in questa pagina sono quelle con la classe 'b1' o 'b2'
        rows = data_table.find_all('tr', class_=['b1', 'b2'])

        if not rows:
            # Se le righe b1/b2 non si trovano (ad esempio, se il modello è GFS che non usa b1/b2)
            # Dobbiamo trovare la riga corretta. Nel tuo HTML non c'è una riga b1/b2.
            # Dobbiamo considerare tutte le righe <tr>, escludendo le prime e le ultime.
            
            # Dal tuo HTML di esempio, il vento (knots) è nella riga 2 (indice 1)
            # e la direzione è nella riga 4 (indice 3).
            
            # Poiché stiamo già cercando l'intestazione, sappiamo che la riga successiva
            # contiene i dati della velocità del vento, e la riga dopo ancora la direzione.
            
            # Questo è più complesso, quindi torniamo all'analisi cella per cella.
            
            # ** RILETTURA DELLO SCHEMA DELLA PAGINA CON IL TUO HTML **
            # La tabella ha le seguenti righe:
            # [0] (fr14h, fr15h, ecc. - Intestazioni orarie)
            # [1] Wind speed (knots)
            # [2] Wind gusts (knots)
            # [3] Wind direction
            # [4] Temperature
            # ... e così via.
            
            # Quindi, la velocità del vento è la riga successiva (indice 1) e la direzione è indice 3.
            
            all_rows = data_table.find_all('tr')
            
            if len(all_rows) < 4:
                print("Errore: Non ci sono abbastanza righe nella tabella per estrarre i dati.")
                return

            # Estrai le righe specifiche
            time_row = all_rows[0].find_all(['td', 'th'])[1:] # Escludi la prima cella "Time"
            speed_row = all_rows[1].find_all(['td', 'th'])[1:]
            direction_row = all_rows[3].find_all(['td', 'th'])[1:]
            
            if not (len(time_row) == len(speed_row) == len(direction_row)):
                print("Errore: Le righe di dati hanno lunghezze diverse.")
                return

            print("--- Previsioni del Vento Favorevoli (S > 10 nodi) ---\n")
            trovato_vento_favorevole = False
            
            for i in range(len(time_row)):
                time_data = time_row[i].get_text(strip=True).replace('\n', ' ')
                wind_speed_str = speed_row[i].get_text(strip=True)
                
                # La direzione è dentro un tag <img>, usiamo l'attributo alt
                # Esempio: <img src="..." alt="WSW"/>
                direction_img = direction_row[i].find('img')
                wind_direction = direction_img['alt'] if direction_img and 'alt' in direction_img.attrs else 'N/D'

                try:
                    wind_speed = float(wind_speed_str)
                except ValueError:
                    continue
                
                # Controlla il criterio: Vento da Sud (S, SSE, SSW, SE, SW) E velocità > 10 nodi
                # N.B.: Ho incluso anche SSE e SSW che sono vicini al Sud.
                is_south = wind_direction in ['S', 'SSW', 'SSE', 'SW', 'SE']

                if is_south and wind_speed > 10:
                    print(f"Data/Ora: {time_data}, Vento: {wind_speed_str} nodi, Direzione: {wind_direction}")
                    trovato_vento_favorevole = True
            
            if not trovato_vento_favorevole:
                print("Nessuna previsione trovata con vento da Sud o Sud-Ovest/Est (> 10 nodi) per i prossimi giorni.")

    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta HTTP (problema di rete o connessione): {e}")
    except Exception as e:
        print(f"Errore durante l'elaborazione dei dati: {e}")

# Esegui la funzione
if __name__ == "__main__":
    fetch_wind_data()