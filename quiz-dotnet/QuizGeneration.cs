using System.Numerics.Tensors;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace QuizDotnet;

partial class Program
{
    // Builds the generation prompt for one chunk, for either a multiple-choice or a true/false question.
    static string BuildQuizPrompt(string chunkContent, string bloomLevel, bool isTrueFalse)
    {
        string common = $@"REQUISITOS:
- Nivel de Taxonomía de Bloom objetivo: {bloomLevel}
- La respuesta correcta debe poder verificarse y justificarse de manera directa y factual con el contenido dado.
- No generes preguntas sobre detalles triviales, saludos, o anécdotas personales.
- No inventes información externa, pero entiende el contexto si la transcripción está mal.
- Ten en cuenta que la transcripcion puede tener ruido o conceptos que no se interpretaron de forma exacta.";

        if (isTrueFalse)
        {
            return $@"Eres un experto en diseño instruccional universitario.
Basándote ÚNICAMENTE en el siguiente contenido de clase, genera UNA pregunta de tipo VERDADERO/FALSO.

{common}
- Redacta una afirmación clara que sea inequívocamente verdadera o falsa según el contenido.
- Las opciones deben ser exactamente ""Verdadero"" y ""Falso"".
- ""correct_option"" debe ser ""a"" si la afirmación es verdadera, ""b"" si es falsa.

Responde únicamente con un objeto JSON válido que siga exactamente esta estructura:
{{
  ""question"": ""afirmación a evaluar..."",
  ""question_type"": ""true_false"",
  ""options"": {{
    ""a"": ""Verdadero"",
    ""b"": ""Falso""
  }},
  ""correct_option"": ""a"",
  ""bloom_level"": ""{bloomLevel}"",
  ""justification"": ""justificación basada en el texto...""
}}

EJEMPLO de una pregunta bien formada (referencia de calidad y estilo, NO copies su contenido):
{{
  ""question"": ""El bounded context define las fronteras de un microservicio, delimitando su ámbito de operaciones y reglas de negocio."",
  ""question_type"": ""true_false"",
  ""options"": {{
    ""a"": ""Verdadero"",
    ""b"": ""Falso""
  }},
  ""correct_option"": ""a"",
  ""bloom_level"": ""comprender"",
  ""justification"": ""Durante la clase se explica que el bounded context delimita las fronteras del microservicio, por lo que la afirmación es verdadera.""
}}

CONTENIDO DE CLASE:
""{chunkContent}""

";
        }

        return $@"Eres un experto en diseño instruccional universitario.
Basándote ÚNICAMENTE en el siguiente contenido de clase, genera UNA pregunta de opción múltiple con 4 opciones (a, b, c, d).

{common}
- Cada pregunta debe tener 3 distractores que vayan acorde al tema.

Responde únicamente con un objeto JSON válido que siga exactamente esta estructura:
{{
  ""question"": ""texto de la pregunta..."",
  ""question_type"": ""multiple_choice"",
  ""options"": {{
    ""a"": ""opción a..."",
    ""b"": ""opción b..."",
    ""c"": ""opción c...""
    ""d"": ""opción d...""
  }},
  ""correct_option"": ""a"",
  ""bloom_level"": ""{bloomLevel}"",
  ""justification"": ""justificación basada en el texto...""
}}

EJEMPLO de una pregunta bien formada (úsalo como referencia de calidad y estilo, NO copies su contenido):
{{
  ""question"": ""¿Qué es el bounded context en el contexto de los microservicios según la explicación dada?"",
  ""question_type"": ""multiple_choice"",
  ""options"": {{
    ""a"": ""Es el límite definido que se establece alrededor del microservicio para identificar su ámbito de operaciones y reglas de negocio."",
    ""b"": ""Es una analogía para describir cómo funciona una cinta magnética en la computación."",
    ""c"": ""Se refiere a las funciones adicionales que un microservicio puede tener además de procesar archivos."",
    ""d"": ""Es el lenguaje de programación específico utilizado dentro del microservicio.""
  }},
  ""correct_option"": ""a"",
  ""bloom_level"": ""comprender"",
  ""justification"": ""La respuesta correcta se basa en la definición directa dada durante la clase, donde se menciona que 'es importante entender el bounding context. Entender las fronteras del microservicio que van a extirpar'. Esto coincide con la opción 'a' que define el bounded context como el límite definido alrededor del microservicio.""
}}

CONTENIDO DE CLASE:
""{chunkContent}""

";
    }

    static async Task<List<QuizQuestion>> GenerateQuizQuestionsAsync(string courseId, int numQuestions, string bloomLevel, string collectionName)
    {
        IChatClient chatClient = new OllamaChatClient(new Uri(OllamaUrl), LlmModel);
        IChatClient judgeClient = new OllamaChatClient(new Uri(OllamaUrl), JudgeModel);
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
            new OllamaEmbeddingGenerator(new Uri(OllamaUrl), EmbeddingModel);
        using var qdrantClient = new QdrantClient(QdrantHost, QdrantPort);

        // Retrieve course chunks from Qdrant
        Console.WriteLine("Retrieving academic chunks from Qdrant for RAG context...");
        // Filter search for this course_id and ACADEMICO category
        var searchFilter = new Filter();
        searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "course_id", Match = new Match { Text = courseId } } });
        searchFilter.Must.Add(new Condition { Field = new FieldCondition { Key = "category", Match = new Match { Text = "ACADEMICO" } } });

        // Bumped scroll limit so 2h-class chunk counts (often 60-150+) are not truncated.
        var scrollResult = await qdrantClient.ScrollAsync(
            collectionName: collectionName,
            filter: searchFilter,
            limit: 500
        );

        var retrievedChunks = scrollResult.Result;
        if (retrievedChunks.Count == 0)
        {
            Console.WriteLine("No academic chunks found in Qdrant.");
            return new List<QuizQuestion>();
        }

        // Adaptive target: stays dynamic with content size but biased smaller (see consts).
        // Caller can override by passing numQuestions > 0.
        int targetQuestions = numQuestions > 0
            ? numQuestions
            : Math.Clamp((int)Math.Ceiling(retrievedChunks.Count / QuestionsPerChunkDivisor), MinQuestions, MaxQuestions);
        Console.WriteLine($"Academic chunks retrieved: {retrievedChunks.Count}. Target questions: {targetQuestions}.");

        // ---- Phase 0: keep only the most significant chunks (classifier confidence × concept density). ----
        // Generating from every chunk is the dominant time sink and dilutes quality; the densest chunks
        // hold the actual concepts, so we generate from those only.
        var candidates = retrievedChunks
            .Select(p => new
            {
                Content = p.Payload.TryGetValue("content", out var contentVal) ? contentVal.StringValue : "",
                Confidence = p.Payload.TryGetValue("confidence_score", out var confVal) ? confVal.DoubleValue : 0.8
            })
            .Where(x => !string.IsNullOrWhiteSpace(x.Content))
            .Select(x => new
            {
                x.Content,
                x.Confidence,
                WordCount = x.Content.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length
            })
            .ToList();

        // Skip thin chunks (they yield trivial questions); relax only if the filter removes everything.
        var dense = candidates.Where(x => x.WordCount >= 30).ToList();
        if (dense.Count == 0) dense = candidates;

        var selectedChunks = dense
            .OrderByDescending(x => x.Confidence * x.WordCount)
            .Take(targetQuestions)
            .Select(x => x.Content)
            .ToList();
        Console.WriteLine($"Selected {selectedChunks.Count} significant chunks for question generation.");

        // ---- Phase 1: generate every question first, so the generator model stays resident (no per-question swap). ----
        var quizOptions = new ChatOptions
        {
            ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(QuizQuestion))
        };
        var candidateQuestions = new List<(QuizQuestion Question, string Source)>();
        // Sprinkle true/false questions across the set (every Nth) per TrueFalseRatio; rest are multiple choice.
        int tfEvery = TrueFalseRatio > 0 ? Math.Max(2, (int)Math.Round(1.0 / TrueFalseRatio)) : int.MaxValue;
        for (int i = 0; i < selectedChunks.Count; i++)
        {
            string chunkContent = selectedChunks[i];
            bool isTrueFalse = tfEvery != int.MaxValue && (i + 1) % tfEvery == 0;
            string quizPrompt = BuildQuizPrompt(chunkContent, bloomLevel, isTrueFalse);

            try
            {
                var quizResponse = await chatClient.GetResponseAsync(quizPrompt, quizOptions);
                var question = JsonSerializer.Deserialize<QuizQuestion>(quizResponse.Text);
                if (question != null && !string.IsNullOrWhiteSpace(question.Question))
                {
                    question.QuestionType = isTrueFalse ? "true_false" : "multiple_choice";
                    candidateQuestions.Add((question, chunkContent));
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating question: {ex.Message}");
            }
        }

        // ---- Phase 2: judge every candidate in one pass (judge model loads once, not once per question). ----
        var valOptions = new ChatOptions
        {
            ResponseFormat = ChatResponseFormat.ForJsonSchema(typeof(ValidationResult))
        };
        var vettedQuestions = new List<QuizQuestion>();
        foreach (var (question, source) in candidateQuestions)
        {
            // Render whatever options exist (2 for true/false, 4 for multiple choice).
            string optionsText = string.Join("\n", question.Options.Select(o => $"{o.Key}) {o.Value}"));

            string valPrompt = $@"Eres un evaluador de preguntas de examen universitario. Tu objetivo es juzgar la calidad y fidelidad factual de la pregunta generada a partir de un fragmento de clase y contrasta con el conocimiento que tengas del tema.

FRAGMENTO DE CLASE:
""{source}""

PREGUNTA EVALUADA ({question.QuestionType}):
Pregunta: {question.Question}
Opciones:
{optionsText}
Respuesta correcta: {question.CorrectOption}

REGLAS DE EVALUACIÓN:
1. Grounding factual (0.0 a 1.0): ¿La respuesta correcta está totalmente respaldada y demostrada de forma directa por el fragmento de clase? Si requiere suposiciones externas, dale puntaje bajo.
2. Calidad de distractores (0.0 a 1.0): ¿Las demás opciones son plausibles pero indiscutiblemente falsas según el fragmento? (Para verdadero/falso, evalúa si la afirmación es inequívoca.)
3. Relevancia (0.0 a 1.0): ¿La pregunta evalúa conceptos clave y no detalles insignificantes?

Calcula el promedio general de estas 3 reglas como un valor entre 0.0 y 1.0.
Responde únicamente con un objeto JSON válido con este formato:
{{""score"": 0.85, ""reason"": ""explicación breve de la evaluación""}}";

            double valScore = 0.0;
            try
            {
                var valResponse = await judgeClient.GetResponseAsync(valPrompt, valOptions);
                var valResult = JsonSerializer.Deserialize<ValidationResult>(valResponse.Text);
                valScore = valResult?.Score ?? 0.0;
                Console.WriteLine($"LLM-as-a-judge score: {valScore} (Reason: {valResult?.Reason})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error validating question: {ex.Message}");
            }

            if (valScore < 0.75)
            {
                Console.WriteLine("Discarding question due to low validation score.");
                continue;
            }

            vettedQuestions.Add(question);
        }

        // ---- Phase 3: drop near-duplicate questions. Embed per item with a guard: bge-m3 emits a NaN
        // vector for some inputs, which Ollama cannot serialize and would otherwise kill a batch call. ----
        var generatedQuestions = new List<QuizQuestion>();
        var keptEmbeddings = new List<Embedding<float>>();
        foreach (var question in vettedQuestions)
        {
            Embedding<float>? qEmb = null;
            try
            {
                var res = await embeddingGenerator.GenerateAsync(new[] { question.Question });
                qEmb = res[0];
            }
            catch (Exception ex)
            {
                // Keep the (already judge-vetted) question; just can't dedup-check this one.
                Console.WriteLine($"Dedup embedding failed, accepting without duplicate check: {ex.Message}");
            }

            if (qEmb != null)
            {
                bool isDuplicate = false;
                for (int j = 0; j < keptEmbeddings.Count; j++)
                {
                    if (TensorPrimitives.CosineSimilarity(qEmb.Vector.Span, keptEmbeddings[j].Vector.Span) > 0.92f)
                    {
                        isDuplicate = true;
                        break;
                    }
                }

                if (isDuplicate)
                {
                    Console.WriteLine("Discarding question as a semantic duplicate of an existing question.");
                    continue;
                }

                keptEmbeddings.Add(qEmb);
            }

            generatedQuestions.Add(question);
            Console.WriteLine($"Accepted question: {question.Question}");
        }

        return generatedQuestions;
    }

    static void SaveQuiz(string inputPath, List<QuizQuestion> questions, string transcriptModel, TimeSpan quizElapsed, TimeSpan totalElapsed)
    {
        string dir = "quizzes";
        Directory.CreateDirectory(dir);
        string baseName = Path.GetFileNameWithoutExtension(inputPath);
        string fileName = $"{baseName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string path = Path.Combine(dir, fileName);

        var output = new
        {
            generated_at = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            source = baseName,
            models = new
            {
                transcript = transcriptModel,
                processing = LlmModel,
                embedding = EmbeddingModel,
                llm_as_judge = JudgeModel
            },
            timings = new
            {
                quiz_generation = FormatElapsed(quizElapsed),
                quiz_generation_seconds = Math.Round(quizElapsed.TotalSeconds, 2),
                pipeline_total = FormatElapsed(totalElapsed),
                pipeline_total_seconds = Math.Round(totalElapsed.TotalSeconds, 2)
            },
            questions
        };

        var json = JsonSerializer.Serialize(output, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });
        File.WriteAllText(path, json);

        Console.WriteLine($"Quiz saved to {path}");
    }
}
