function simulatePrediction() {
  const input = document.getElementById('fileInput');
  const result = document.getElementById('predictionResult');
  if (!input || !input.files.length) {
    result.textContent = 'Veuillez sélectionner un fichier CSV ou WAV.';
    return;
  }
  const probability = (80 + Math.random() * 19).toFixed(1);
  result.textContent = `Parkinson detected — ${probability}%`;
}
