# Plan de implementación: Quiz Generator con IA (.NET 10 Unified Stack)
**Stack:** Monolito modular C# (.NET 10) + Angular + PostgreSQL + Docker  
**Integración IA:** Procesamiento nativo en .NET 10 (sin microservicio Python)  
**Equipo:** 8 personas · 4h/día · 50 días hábiles · 1600h totales

---

## Arquitectura general (.NET 10 Unified Stack)

```
Angular
   │ HTTP
   ▼
Monolito C# (.NET 10)
   ├── Módulo Cursos / Auth / Users     (existente)
   ├── Módulo Quizzes                   (nuevo)
   └── Módulo Pipeline IA               (nuevo)
          ├── Hangfire                  (cola de jobs en segundo plano)
          ├── Whisper.net               (transcripción local de audio/video)
          ├── ML.NET / NAudio           (VAD y diarización de voces nativa)
          ├── Microsoft.Extensions.AI   (abstracción de Ollama y embeddings)
          ├── TensorPrimitives          (SIMD-optimizado para chunking semántico)
          └── Qdrant.Client             (base de datos vectorial)
```

**Principio clave:** Todo el pipeline de procesamiento de IA se ejecuta dentro del monolito C# en .NET 10. Se elimina por completo el microservicio Python, simplificando la infraestructura y reduciendo la latencia de red. El monolito orquesta Hangfire para tareas en segundo plano y llama a Ollama y Qdrant localmente por HTTP/gRPC.

---

## Sprint 1 — Infraestructura base y transcripción nativa (.NET 10)
**Días 1–10 · 320h · 8 personas**

### Objetivo
Video/audio entra al sistema → transcript con timestamps y speakers sale en PostgreSQL de forma nativa desde .NET 10 sin dependencias externas de Python.

### Tareas

| # | Tarea | Responsable | Días est. |
|---|-------|-------------|-----------|
| 1 | Docker Compose: monolito C# (.NET 10) + Ollama + Qdrant + MinIO + PostgreSQL | DevOps | 2 |
| 2 | Estructuración del Módulo Pipeline IA en el monolito C# e inyección de dependencias | Backend C# | 2 |
| 3 | Configuración de Hangfire sobre la base de datos de Postgres existente | Backend C# | 1 |
| 4 | Endpoint `POST /api/pipeline/sources`: recibe archivo en monolito y encola el job | Backend C# | 1 |
| 5 | Integrador de Whisper.net: Carga de modelos GGML y transcripción local en segundo plano | Backend C# | 2 |
| 6 | Integrador de NAudio + ML.NET para la diarización de voces nativa (VAD y KMeans) | Backend C# | 2 |
| 7 | Esquema Postgres: tablas `sources`, `transcripts`, `chunks`, `jobs`, `audit_log` | DB | 1 |
| 8 | MinIO: almacenamiento local de los archivos de video y audio originales | DevOps | 1 |
| 9 | Tests de integración: subida de archivos → transcripción y segmentación en Postgres | QA | 2 |

### Esquema de base de datos (nuevas tablas)

```sql
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256
    file_name VARCHAR(500),
    course_id UUID,
    duration_seconds INT,
    status VARCHAR(50),                       -- pending, processing, done, failed
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES sources(id),
    raw_json JSONB NOT NULL,                  -- output estructurado de Whisper.net
    language VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES sources(id),
    content TEXT NOT NULL,
    topic_summary VARCHAR(500),
    speaker VARCHAR(50),
    confidence_score FLOAT,
    category VARCHAR(50),                     -- ACADEMICO, OFF_TOPIC, ERROR, ANECDOTA
    ts_start FLOAT,
    ts_end FLOAT,
    bloom_level VARCHAR(50),
    qdrant_vector_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Sprint 2 — Pipeline de limpieza y chunking semántico en .NET 10
**Días 11–20 · 320h · 8 personas**

### Objetivo
Transcript crudo → chunks limpios con metadata semántica en Postgres usando `Microsoft.Extensions.AI` y `TensorPrimitives`.

### Tareas

| # | Tarea | Responsable | Días est. |
|---|-------|-------------|-----------|
| 1 | Configurar clientes de IA locales (`OllamaEmbeddingGenerator` y `OllamaChatClient`) | Backend C# | 1 |
| 2 | Implementación de división semántica nativa usando `TensorPrimitives.CosineSimilarity` | Backend C# | 2 |
| 3 | Clasificador de relevancia académica con LLM (`Microsoft.Extensions.AI.ChatOptions`) | Backend C# | 2 |
| 4 | Sistema de auditoría y descarte persistido en base de datos Postgres | Backend C# | 1 |
| 5 | Summarizer temático por chunk usando llamadas a Ollama con ResponseSchema | Backend C# | 2 |
| 6 | Dashboard en Angular: visualización de chunks indexados, filtrados y auditoría | Frontend | 3 |
| 7 | Calibración de prompts de clasificación del LLM local en español | Backend/QA | 2 |
| 8 | Tests unitarios y de integración de limpieza y segmentación semántica | QA | 1 |

### Configuración del Chunker Semántico (.NET 10)

```csharp
using System.Numerics.Tensors;
using Microsoft.Extensions.AI;

public class SemanticChunker
{
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private const float SimilarityThreshold = 0.5f;

    public async Task<List<TextChunk>> ChunkTextAsync(List<string> sentences)
    {
        var chunks = new List<TextChunk>();
        var embeddings = await _embeddingGenerator.GenerateAsync(sentences);
        
        var currentChunk = new List<string> { sentences[0] };
        
        for (int i = 0; i < sentences.Count - 1; i++)
        {
            ReadOnlySpan<float> vecA = embeddings[i].Vector.Span;
            ReadOnlySpan<float> vecB = embeddings[i + 1].Vector.Span;
            
            float similarity = TensorPrimitives.CosineSimilarity(vecA, vecB);
            
            if (similarity < SimilarityThreshold)
            {
                chunks.Add(new TextChunk(string.Join(" ", currentChunk)));
                currentChunk.Clear();
            }
            currentChunk.Add(sentences[i + 1]);
        }
        
        if (currentChunk.Any())
        {
            chunks.Add(new TextChunk(string.Join(" ", currentChunk)));
        }
        
        return chunks;
    }
}
```

---

## Sprint 3 — Base de conocimiento y vectores en Qdrant
**Días 21–30 · 320h · 8 personas**

### Objetivo
Indexar embeddings de chunks en Qdrant usando el cliente oficial gRPC y posibilitar la búsqueda semántica.

### Tareas

| # | Tarea | Responsable | Días est. |
|---|-------|-------------|-----------|
| 1 | Integración del paquete `Qdrant.Client` en el proyecto de infraestructura C# | Backend C# | 1 |
| 2 | Inicialización de colecciones en Qdrant con configuración de 768d y distancia coseno | Backend C# | 1 |
| 3 | Mapeo y conversión de payloads estructurados para los vectores en Qdrant | Backend C# | 2 |
| 4 | Desarrollo del servicio de búsqueda vectorial `GET /api/kb/search` en el controlador | Backend C# | 2 |
| 5 | Servicio de invalidación semántica por `source_id` (eliminación física de vectores) | Backend C# | 1 |
| 6 | Configuración de Docker volumes persistentes para la base de datos Qdrant | DevOps | 1 |
| 7 | Implementación del buscador y explorador de conocimiento en la interfaz Angular | Frontend | 3 |
| 8 | Pruebas de velocidad de recuperación semántica en base de datos (Umbral: <300ms) | QA | 1 |

---

## Sprint 4 — Agente de Quizzes y LLM-as-a-Judge (.NET 10)
**Días 31–40 · 320h · 8 personas**

### Objetivo
Generar cuestionarios educativos usando RAG con validación automatizada de grounding factual.

### Tareas

| # | Tarea | Responsable | Días est. |
|---|-------|-------------|-----------|
| 1 | Pipeline de RAG: búsqueda en Qdrant → inyección en prompt en el backend C# | Backend C# | 2 |
| 2 | Formulación estructurada de preguntas con taxonomía de Bloom usando `ChatOptions` | Backend C# | 2 |
| 3 | Validador LLM-as-a-Judge para evaluar grounding y calidad de distractores | Backend C# | 2 |
| 4 | Filtro de duplicados semánticos calculando similitud coseno contra histórico de sesión | Backend C# | 1 |
| 5 | APIs CRUD de gestión de Quizzes (Controladores ASP.NET Core) | Backend C# | 2 |
| 6 | Integración de las tablas `quizzes` y `quiz_questions` en base de datos PostgreSQL | DB | 1 |
| 7 | Pruebas funcionales de generación: validación manual de calidad del contenido | QA / Docente | 2 |

### Configuración del Agente de Generación (.NET 10)

```csharp
public class QuizAgent
{
    private readonly IChatClient _chatClient;

    public async Task<GeneratedQuestion?> CreateQuestionAsync(string chunkContent, string bloomLevel)
    {
        var options = new ChatOptions
        {
            Temperature = 0.3f,
            ResponseFormat = ChatResponseFormat.Json,
            ResponseSchema = typeof(GeneratedQuestion) // Validación estricta nativa de .NET 10
        };

        var response = await _chatClient.CompleteAsync(
            $"Genera una pregunta de tipo '{bloomLevel}' en base a este fragmento: {chunkContent}", 
            options
        );

        return JsonSerializer.Deserialize<GeneratedQuestion>(response.Message.Text);
    }
}
```

---

## Sprint 5 — UI Angular, exportación de cuestionarios y endurecimiento
**Días 41–50 · 320h · 8 personas**

### Objetivo
Flujo web completo en Angular de extremo a extremo, editor de quizzes, observabilidad y despliegue del demo.

### Tareas

| # | Tarea | Responsable | Días est. |
|---|-------|-------------|-----------|
| 1 | UI Angular: subida de archivos → progreso en tiempo real (SignalR) → quiz generado | Frontend / Backend| 3 |
| 2 | Editor interactivo de preguntas: editar distractores, cambiar Bloom, confirmar | Frontend | 2 |
| 3 | Visualizador gráfico de distribución de niveles Bloom por cuestionario | Frontend | 1 |
| 4 | Exportadores de quizzes a formato H5P (para Moodle) y archivos JSON estándar | Backend C# | 2 |
| 5 | Observabilidad del pipeline: métricas de tiempo por etapa con OpenTelemetry | DevOps | 2 |
| 6 | Endurecimiento de seguridad de las APIs e implementación de Health Checks nativos | DevOps | 1 |
| 7 | Pruebas de regresión con sets de grabaciones reales de larga duración | QA | 1 |
| 8 | Documentación de operación del sistema y manual de usuario | Todos | 1 |
| 9 | Ensayo y despliegue del demo final de producción | Todos | 2 |

---

## Decisiones Técnicas y Parámetros Recomendados

### Modelos y Consumo de Recursos
* **Embeddings:** `nomic-embed-text` (~1GB RAM) — Vectoriza oraciones rápidamente.
* **Inferencia (Chat y Clasificación):** `qwen2.5-coder:1.5b` (~1.5GB RAM) — Permite inferencia local extremadamente ágil y con soporte nativo de JSON Schema de .NET 10 en CPU de la VM de desarrollo.
* **Motor Whisper:** `ggml-tiny.bin` o `ggml-base.bin` vía Whisper.net para transcripción directa de hilos paralelos de CPU.

*Documento actualizado para .NET 10 Unified Stack.*
