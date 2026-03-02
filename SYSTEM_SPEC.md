# SYNDICATE SYSTEM SPEC

## 1. Objetivo

Construir un sistema cuantitativo institucional para NBA con:
- Moneyline calibrado
- Validación walk-forward real
- Control de incertidumbre
- Data Gatekeeper independiente
- Benchmark contra mercado

No optimizar por accuracy.
Optimizar por LogLoss, Brier y estabilidad temporal.

---

## 2. Arquitectura Actual (V12)

### Modelo Principal
- Output: win_logits
- Loss: BCEWithLogitsLoss
- Sin calibración dentro del modelo

### Calibración
- IsotonicRegression post-hoc
- Ajustada sobre logits OOS del walk-forward

### Walk-Forward
- 3 folds expandibles
- Scaler independiente por fold
- Logits OOS pooled para calibración

### Incertidumbre
- BrierErrorPredictor
- Target: (p_calibrated - y)^2
- Softplus output
- No modifica probabilidad base

### Gatekeeper
- Separado del modelo
- Valida:
  - Integridad temporal
  - NaN / Inf
  - Separación de folds
  - Mínimos de muestra

---

## 3. Métricas Oficiales

Primarias:
- LogLoss
- Brier Score
- Calibration slope
- ECE
- AUC

Secundarias:
- Accuracy
- Profit factor
- Sharpe

---

## 4. Reglas de Gobernanza

- No modificar arquitectura sin justificación cuantitativa.
- No mezclar calibración dentro del modelo.
- No introducir nuevas features sin validación.
- No usar CV aleatorio.
- Todo cambio importante debe hacerse en rama feature/*.

---

## 5. Roadmap

Fase 1: Validación estadística profunda ML.
Fase 2: Margin model institucional.
Fase 3: Totals model.
Fase 4: EV engine + sizing.
Fase 5: Drift detection automatizado.