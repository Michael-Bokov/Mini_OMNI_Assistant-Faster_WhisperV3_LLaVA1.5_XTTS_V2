const recordBtn = document.getElementById("recordBtn");
const sendBtn = document.getElementById("sendBtn");
const statusDiv = document.getElementById("status");
const imageInput = document.getElementById("imageInput");
const languageSelect = document.getElementById("languageSelect");
const promptInput = document.getElementById("promptInput");
const messages = document.getElementById("messages");
const responseAudio = document.getElementById("responseAudio");

let mediaRecorder = null;
let audioChunks = [];
let recordedAudioBlob = null;
let streamRef = null;
let isRecording = false;

function setStatus(text, isError = false) {
    statusDiv.textContent = text;
    statusDiv.classList.toggle("error", isError);
}

function setControlsBusy(busy) {
    sendBtn.disabled = busy || isRecording;
    imageInput.disabled = busy;
    languageSelect.disabled = busy;
    promptInput.disabled = busy;
    recordBtn.disabled = busy;
}

function setUserBubbleContent(bubble, text, metaText = "") {
    bubble.textContent = "";
    bubble.appendChild(document.createTextNode(text));
    if (metaText) {
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = metaText;
        bubble.appendChild(meta);
    }
    messages.scrollTop = messages.scrollHeight;
}

function appendAssistantMessage(text) {
    const bubble = document.createElement("div");
    bubble.className = "msg assistant";
    bubble.textContent = text;
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
}

function safeStopStream() {
    if (streamRef) {
        streamRef.getTracks().forEach((track) => track.stop());
        streamRef = null;
    }
}

async function toggleRecording() {
    if (!isRecording) {
        try {
            streamRef = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(streamRef);
            audioChunks = [];
            recordedAudioBlob = null;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                recordedAudioBlob = new Blob(audioChunks, { type: "audio/webm" });
                audioChunks = [];
                safeStopStream();
                setStatus("Голос записан. Нажмите «Отправить».");
            };

            mediaRecorder.start();
            isRecording = true;
            recordBtn.textContent = "Остановить запись";
            recordBtn.classList.add("danger");
            setStatus("Идёт запись…");
            sendBtn.disabled = true;
        } catch (error) {
            setStatus("Нет доступа к микрофону.", true);
        }
        return;
    }

    mediaRecorder.stop();
    isRecording = false;
    recordBtn.textContent = "Начать запись";
    recordBtn.classList.remove("danger");
    sendBtn.disabled = false;
}

function imageMetaNote(hasImage) {
    return hasImage ? "Изображение прикреплено" : "";
}

async function sendMessage() {
    const prompt = promptInput.value.trim();
    const imageFile = imageInput.files[0] || null;
    const language = languageSelect.value;

    if (!prompt && !recordedAudioBlob) {
        setStatus("Введите текст или запишите голос.", true);
        return;
    }

    const hasVoice = Boolean(recordedAudioBlob);
    const hasImage = Boolean(imageFile);

    const userBubble = document.createElement("div");
    userBubble.className = "msg user";
    if (hasVoice && !prompt) {
        setUserBubbleContent(userBubble, "Распознаю речь…", imageMetaNote(hasImage));
    } else {
        setUserBubbleContent(
            userBubble,
            prompt || "…",
            hasVoice && prompt ? "Будет добавлена голосовая расшифровка" : imageMetaNote(hasImage)
        );
    }
    messages.appendChild(userBubble);
    messages.scrollTop = messages.scrollHeight;

    setControlsBusy(true);
    setStatus("Обрабатываю запрос…");

    const formData = new FormData();
    formData.append("language", language);
    if (prompt) formData.append("prompt", prompt);
    if (imageFile) formData.append("image", imageFile);
    if (recordedAudioBlob) {
        formData.append("audio", recordedAudioBlob, "recording.webm");
    }

    try {
        const response = await fetch("/chat", {
            method: "POST",
            body: formData,
        });

        const raw = await response.text();
        let payload = null;
        if (raw) {
            try {
                payload = JSON.parse(raw);
            } catch {
                throw new Error(
                    raw.trim().slice(0, 400) ||
                        `Сервер вернул не JSON (HTTP ${response.status}).`
                );
            }
        }

        if (!response.ok) {
            const detail = payload && payload.detail;
            const msg =
                typeof detail === "string"
                    ? detail
                    : Array.isArray(detail)
                      ? detail.map((d) => d.msg || d).join(" ")
                      : raw?.trim().slice(0, 400) ||
                        `Ошибка HTTP ${response.status}`;
            throw new Error(msg);
        }

        const shown = (payload.user_text || "").trim();
        setUserBubbleContent(
            userBubble,
            shown || "(пустой запрос)",
            imageMetaNote(hasImage)
        );

        appendAssistantMessage(payload.assistant_text);

        if (payload.audio_url) {
            responseAudio.src = payload.audio_url;
            await responseAudio.play().catch(() => {});
        }

        setStatus("Готово.");
        promptInput.value = "";
        imageInput.value = "";
        recordedAudioBlob = null;
    } catch (error) {
        setUserBubbleContent(
            userBubble,
            "Не удалось отправить.",
            error.message || ""
        );
        appendAssistantMessage("Повторите попытку или проверьте сервисы ASR / VLM / TTS.");
        setStatus(error.message || "Сетевая ошибка.", true);
    } finally {
        setControlsBusy(false);
    }
}

recordBtn.addEventListener("click", toggleRecording);
sendBtn.addEventListener("click", sendMessage);
promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        sendMessage();
    }
});
