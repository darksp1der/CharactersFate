const predictBtn = document.getElementById("predict-btn");
const statusBox = document.getElementById("status");
const confBox = document.getElementById("confidence");
const metricsBtn = document.getElementById("show-metrics");
const reasonBtn = document.getElementById("show-reason");
const metricsBox = document.getElementById("extra-metrics");
const reasonBox = document.getElementById("reason");

let cache = {};

predictBtn.addEventListener("click", () => {
  const text = document.getElementById("history").value.trim();
  if (!text) return alert("Please enter a character history!");

  statusBox.textContent = "⏳ Predicting...";
  confBox.textContent = "";
  metricsBox.style.display = "none";
  reasonBox.style.display = "none";
  metricsBtn.style.display = "none";
  reasonBtn.style.display = "none";

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  })
    .then(res => res.json())
    .then(data => {
      cache = data;
      statusBox.textContent = data.status === "Alive" ? "🟢 Status: Alive" : "🔴 Status: Dead";
      confBox.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
      metricsBtn.style.display = "inline-block";
      reasonBtn.style.display = "inline-block";
    });
});

let metricsVisible = false;

metricsBtn.addEventListener("click", () => {
  metricsVisible = !metricsVisible;

  if (metricsVisible) {
    metricsBox.style.display = "block";
    metricsBox.innerHTML = `
      🔢 Entropy: ${cache.entropy.toFixed(4)}<br>
      📉 Margin: ${cache.margin.toFixed(4)}<br>
      📤 Logits: ${cache.logits.map(x => x.toFixed(3)).join(", ")}
    `;
    metricsBtn.textContent = "🙈 Hide Metrics";
  } else {
    metricsBox.style.display = "none";
    metricsBtn.textContent = "📊 Show Metrics";
  }
});

reasonBtn.addEventListener("click", () => {
  reasonBox.style.display = "block";
  reasonBox.textContent = "💭 Generating explanation...";

  fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: cache.text, status: cache.status })
  })
    .then(res => res.json())
    .then(data => {
      reasonBox.textContent = `💡 Reasoning: ${data.explanation}`;
    });
});
