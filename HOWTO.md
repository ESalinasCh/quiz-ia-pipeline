# Guía de Ejecución y Pruebas (HOWTO)

Este documento contiene las instrucciones paso a paso para configurar, compilar y ejecutar tanto el POC de Python como la aplicación unificada en C# (.NET 10).

---

## 🛠️ Instalación de Dependencias del Sistema (Ambiente Limpio)

Si vas a ejecutar esto en un entorno Ubuntu/Debian completamente nuevo y limpio, sigue estos comandos en orden para instalar todas las herramientas necesarias:

### 1. Actualizar el Sistema e Instalar FFmpeg y Curl
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg curl git
```

### 2. Instalar el SDK de .NET 10
Descarga e instala el SDK de .NET 10 en tu directorio de usuario:
```bash
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 10.0

# Agregar .NET al PATH en tu sesión actual de terminal
export PATH="$HOME/.dotnet:$PATH"
# O hazlo persistente agregándolo a tu shell profile:
echo 'export PATH="$HOME/.dotnet:$PATH"' >> ~/.bashrc
```

### 3. Instalar Docker y Docker Compose
```bash
# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Añadir tu usuario al grupo docker para ejecutarlo sin sudo (opcional, requiere reiniciar sesión)
sudo usermod -aG docker $USER
```

### 4. Instalar Ollama (Servidor de Modelos IA)
Ollama correrá nativamente en el host:
```bash
# Descargar e instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 🔌 Preparación de Servicios Comunes (Host)

Ambos pipelines requieren una base de datos vectorial Qdrant activa y las imágenes del modelo cargadas en el servidor Ollama local.

### 1. Iniciar la Base de Datos Vectorial Qdrant
Qdrant se ejecuta en un contenedor Docker con volúmenes persistentes. Para levantarlo:
```bash
cd /home/ubuntu/quiz/poc-audio-processing
docker compose up -d qdrant
```
Esto habilitará el puerto HTTP de Qdrant en `127.0.0.1:6333` y el puerto gRPC en `127.0.0.1:6334`.

### 2. Verificar Modelos en Ollama Host
Asegúrate de tener corriendo Ollama en el host local y que cuente con los modelos correctos:
```bash
curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":"[^"]*"'
```
*Debe retornar al menos:*
* `"name":"qwen2.5-coder:1.5b"`
* `"name":"nomic-embed-text:latest"`

Si no los tienes descargados, ejecuta:
```bash
ollama pull qwen2.5-coder:1.5b
ollama pull nomic-embed-text
```

---

## 🐍 Instrucciones para Python POC

El POC de Python está estructurado como un servicio web API REST con FastAPI.

### 1. Levantar el Servicio FastAPI
Construye e inicia el contenedor del servicio de audio:
```bash
cd /home/ubuntu/quiz/poc-audio-processing
docker compose build
docker compose up -d poc-audio-service
```
El servicio estará disponible en `http://127.0.0.1:8000`.

### 2. Ejecutar Pruebas Automatizadas (Python)
Para verificar que el microservicio transcribe, indexa y genera preguntas correctamente, ejecuta el script de prueba integrado:
```bash
python3 verify_service.py
```
Este script sube un archivo WAV de prueba, valida la API de transcripción y solicita la generación de preguntas de cuestionario a los endpoints del contenedor.

### 3. Detener el Servicio
Una vez finalizadas las pruebas de Python, puedes detener los contenedores para liberar recursos:
```bash
docker compose down
```

---

## ⚡ Instrucciones para C# (.NET 10)

La versión de C# se ejecuta de forma nativa directamente en la máquina virtual para optimizar el rendimiento y acelerar el debug.

### 1. Compilar el Proyecto
Dirígete a la carpeta del proyecto y compílalo usando el SDK de .NET 10:
```bash
cd /home/ubuntu/quiz/quiz-dotnet
/home/ubuntu/.dotnet/dotnet build
```

### 2. Ejecutar Pipeline con Audio de Prueba (Corto)
Para una comprobación rápida de 20 segundos que verifique la extracción por FFmpeg, la descarga automatizada del modelo Whisper, el cálculo de similitud y la indexación en Qdrant, ejecuta:
```bash
/home/ubuntu/.dotnet/dotnet run -- /home/ubuntu/quiz/test01_20s.wav
```
*Nota: Al ser una introducción corta, el clasificador catalogará el contenido como `OFF_TOPIC` y generará 0 preguntas académicas (comportamiento esperado).*

### 3. Ejecutar Pipeline con Audio de Clase (Largo)
Para probar la ingesta real de clases académicas y generar el cuestionario de 3 preguntas educativas bajo la taxonomía de Bloom con el validador LLM-as-a-judge activo, ejecuta:
```bash
/home/ubuntu/.dotnet/dotnet run -- /home/ubuntu/quiz/repasosemana2.mp3
```
*Este comando demorará un par de minutos en transcribir y procesar los 21 chunks en la CPU del host, imprimiendo las preguntas validadas finales en la consola al concluir.*
