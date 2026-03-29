export const postAudio = async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "hum.wav");

    try {
        const response = await fetch("http://localhost:8000/", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        return data.prediction;
    } catch (e) {
        console.error("API error", e);
        return "Error";
    }
};