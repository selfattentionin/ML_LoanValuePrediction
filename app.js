const FEATURE_ORDER = [
  "Income",
  "Age",
  "Experience",
  "Married",
  "House_Ownership",
  "Car_Ownership",
  "Profession",
  "CITY",
  "STATE",
  "CURRENT_JOB_YRS",
  "CURRENT_HOUSE_YRS",
];

const NUMERIC_FIELDS = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"];

const predictionText = document.getElementById("predictionText");
const riskScore = document.getElementById("riskScore");
const riskBar = document.getElementById("riskBar");
const reasonText = document.getElementById("reasonText");
const modelStatus = document.getElementById("modelStatus");
const loanValueAmount = document.getElementById("loanValueAmount");
const loanValueHint = document.getElementById("loanValueHint");
const form = document.getElementById("loanForm");

let modelSession = null;
let loanValueSession = null;
let categoryMapping = {};

function formatCurrency(value) {
  const numericValue = Number(value) || 0;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(numericValue);
}

async function loadModelAssets() {
  try {
    const mappingResponse = await fetch("feature_mapping.json");
    if (!mappingResponse.ok) {
      throw new Error("feature_mapping.json not found. Run the training script to generate it.");
    }
    categoryMapping = await mappingResponse.json();

    modelSession = await ort.InferenceSession.create("loan_risk_model.onnx");
    loanValueSession = await ort.InferenceSession.create("loan_value_model.onnx");

    modelStatus.textContent = "Models ready";
    modelStatus.style.borderColor = "rgba(52, 211, 153, 0.35)";
    modelStatus.style.background = "rgba(52, 211, 153, 0.12)";
    modelStatus.style.color = "#34d399";
  } catch (error) {
    modelStatus.textContent = "Models missing";
    modelStatus.style.borderColor = "rgba(248, 113, 113, 0.35)";
    modelStatus.style.background = "rgba(248, 113, 113, 0.12)";
    modelStatus.style.color = "#f87171";
    reasonText.textContent = error.message;
    console.error(error);
  }
}

function getFieldValue(fieldName) {
  const element = document.getElementById(fieldName);
  if (!element) return "";
  return element.value;
}

function encodeInputFromForm() {
  const values = [];

  FEATURE_ORDER.forEach((fieldName) => {
    const rawValue = getFieldValue(fieldName);

    if (NUMERIC_FIELDS.includes(fieldName)) {
      values.push(Number(rawValue || 0));
      return;
    }

    const map = categoryMapping[fieldName] || {};
    const encoded = map[String(rawValue)] ?? 0;
    values.push(Number(encoded));
  });

  return new ort.Tensor("float32", new Float32Array(values), [1, values.length]);
}

function updateRiskVisual(score) {
  const clamped = Math.max(0, Math.min(100, score));
  const label = clamped >= 70 ? "High Risk" : clamped >= 40 ? "Medium Risk" : "Low Risk";

  riskScore.textContent = Math.round(clamped);
  riskBar.style.width = `${clamped}%`;

  const ring = document.querySelector(".ring");
  ring.style.background = `conic-gradient(
    ${clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399"} 0deg,
    ${clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399"} ${clamped * 3.6}deg,
    rgba(148, 163, 184, 0.12) ${clamped * 3.6}deg 360deg
  )`;

  predictionText.textContent = label;
  predictionText.style.color = clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399";

  if (clamped >= 70) {
    reasonText.textContent = "This customer shows a high chance of default and may need stricter underwriting checks.";
  } else if (clamped >= 40) {
    reasonText.textContent = "This is a moderate-risk profile. Additional review may reduce approval risk.";
  } else {
    reasonText.textContent = "This customer appears financially stable and is within a healthy risk range.";
  }
}

function updateLoanValueVisual(amount) {
  const numericAmount = Number(amount) || 0;
  const safeAmount = Math.max(0, Math.min(5000000, numericAmount));

  loanValueAmount.textContent = formatCurrency(safeAmount);
  loanValueHint.textContent = safeAmount > 0 ? "Estimated value based on affordability and profile" : "No estimated value available";
}

function extractProbability(outputs) {
  const outputValues = Object.values(outputs || {});

  if (!outputValues.length) {
    return null;
  }

  const probabilityTensor = outputValues.find((tensor) => {
    const data = tensor && tensor.data ? tensor.data : [];
    return data.length > 1 && Number.isFinite(Number(data[0])) && Number.isFinite(Number(data[1]));
  }) || outputValues[0];

  const data = probabilityTensor && probabilityTensor.data ? probabilityTensor.data : null;
  if (!data || !data.length) {
    return null;
  }

  if (data.length >= 2) {
    return Number(data[1]);
  }

  return Number(data[0]);
}

function extractPredictionValue(outputs) {
  const outputValues = Object.values(outputs || {});

  if (!outputValues.length) {
    return null;
  }

  const firstTensor = outputValues[0];
  const data = firstTensor && firstTensor.data ? firstTensor.data : [];
  if (!data || !data.length) {
    return null;
  }

  return Number(data[0]);
}

async function runPrediction() {
  if (!modelSession) {
    reasonText.textContent = "The ONNX model is not available yet. Run the training notebook first to export the model.";
    return;
  }

  try {
    const inputTensor = encodeInputFromForm();
    const outputs = await modelSession.run({ input: inputTensor });
    window.__lastOnnxOutputs = outputs;
    const probability = extractProbability(outputs);
    window.__lastProbability = probability;

    if (probability === null || Number.isNaN(probability) || !Number.isFinite(probability)) {
      window.__lastPredictionError = new Error("The model returned an invalid probability.");
      reasonText.textContent = "The model returned an invalid probability. Please try again with updated values.";
      return;
    }

    const riskPercent = probability * 100;
    updateRiskVisual(riskPercent);

    if (loanValueSession) {
      const loanValueOutputs = await loanValueSession.run({ input: inputTensor });
      const loanValuePrediction = extractPredictionValue(loanValueOutputs);

      if (loanValuePrediction !== null && Number.isFinite(loanValuePrediction)) {
        updateLoanValueVisual(loanValuePrediction);
      }
    }
  } catch (error) {
    console.error(error);
    window.__lastPredictionError = error;
    reasonText.textContent = error && error.message ? error.message : String(error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPrediction();
});

form.addEventListener("input", async () => {
  await runPrediction();
});

form.addEventListener("change", async () => {
  await runPrediction();
});

loadModelAssets();
