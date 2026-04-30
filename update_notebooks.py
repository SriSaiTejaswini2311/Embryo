import json

def append_to_notebook(filepath, new_cells):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    nb['cells'].extend(new_cells)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

# 1. Update embryo_classification.ipynb
cells_1 = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# SECTION: MODEL VALIDATION\n",
    "\n",
    "This section verifies that the stage classification works correctly on its own by calculating accuracy, visualizing confusion matrices, and inspecting sample predictions and edge cases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "\n",
    "# 1. Extract true labels and predictions\n",
    "y_true = []\n",
    "y_pred_probs = []\n",
    "\n",
    "for images, labels in val_ds:\n",
    "    preds = model.predict(images, verbose=0)\n",
    "    y_pred_probs.extend(preds)\n",
    "    y_true.extend(np.argmax(labels.numpy(), axis=1))\n",
    "\n",
    "y_true = np.array(y_true)\n",
    "y_pred_probs = np.array(y_pred_probs)\n",
    "y_pred_classes = np.argmax(y_pred_probs, axis=1)\n",
    "\n",
    "# 2. Accuracy Metrics\n",
    "acc = accuracy_score(y_true, y_pred_classes)\n",
    "print(f\"Validation Accuracy: {acc:.4f}\")\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_true, y_pred_classes, target_names=class_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Confusion Matrix Visualization\n",
    "cm = confusion_matrix(y_true, y_pred_classes)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)\n",
    "plt.title('Stage Classification Confusion Matrix')\n",
    "plt.ylabel('True Label')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Sample Predictions & Confidence Distribution\n",
    "plt.figure(figsize=(15, 10))\n",
    "for images, labels in val_ds.take(1):\n",
    "    preds = model.predict(images, verbose=0)\n",
    "    for i in range(min(9, len(images))):\n",
    "        plt.subplot(3, 3, i + 1)\n",
    "        img_vis = (images[i].numpy() + 1.0) / 2.0\n",
    "        img_vis = np.clip(img_vis, 0, 1)\n",
    "        plt.imshow(img_vis)\n",
    "        true_class = class_names[np.argmax(labels[i])]\n",
    "        pred_class = class_names[np.argmax(preds[i])]\n",
    "        conf = np.max(preds[i])\n",
    "        color = 'green' if true_class == pred_class else 'red'\n",
    "        plt.title(f\"True: {true_class}\\nPred: {pred_class} ({conf:.2f})\", color=color)\n",
    "        plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Edge Case Testing (Simulated Noise / Low Quality)\n",
    "print(\"\\n--- Edge Case Testing ---\")\n",
    "\n",
    "def add_noise(image):\n",
    "    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.5, dtype=tf.float32)\n",
    "    return tf.clip_by_value(image + noise, -1.0, 1.0)\n",
    "\n",
    "noisy_ds = val_ds.map(lambda x, y: (add_noise(x), y))\n",
    "\n",
    "for images, labels in noisy_ds.take(1):\n",
    "    preds = model.predict(images, verbose=0)\n",
    "    plt.figure(figsize=(10, 4))\n",
    "    for i in range(min(3, len(images))):\n",
    "        plt.subplot(1, 3, i + 1)\n",
    "        img_vis = (images[i].numpy() + 1.0) / 2.0\n",
    "        plt.imshow(np.clip(img_vis, 0, 1))\n",
    "        true_class = class_names[np.argmax(labels[i])]\n",
    "        pred_class = class_names[np.argmax(preds[i])]\n",
    "        conf = np.max(preds[i])\n",
    "        plt.title(f\"Noisy Edge Case\\nPred: {pred_class} ({conf:.2f})\")\n",
    "        plt.axis('off')\n",
    "    plt.show()\n",
    "    break"
   ]
  }
]
append_to_notebook('research/embryo_classification.ipynb', cells_1)

# 2. Update embryo_research_malpani.ipynb
cells_2 = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# SECTION: CLINICAL AUDIT\n",
    "\n",
    "This section verifies the grading model's exact Gardner match rate, per-head metrics, and performs failure analysis on minority classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n",
    "\n",
    "# 1. Extract true labels and predictions (assuming test_loader or val_loader exists)\n",
    "y_true_exp, y_true_icm, y_true_te = [], [], []\n",
    "y_pred_exp, y_pred_icm, y_pred_te = [], [], []\n",
    "\n",
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    for images, labels in val_loader:\n",
    "        images = images.to(device)\n",
    "        out_exp, out_icm, out_te = model(images)\n",
    "        \n",
    "        y_pred_exp.extend(torch.argmax(out_exp, dim=1).cpu().numpy())\n",
    "        y_pred_icm.extend(torch.argmax(out_icm, dim=1).cpu().numpy())\n",
    "        y_pred_te.extend(torch.argmax(out_te, dim=1).cpu().numpy())\n",
    "        \n",
    "        y_true_exp.extend(labels['expansion'].numpy())\n",
    "        y_true_icm.extend(labels['icm'].numpy())\n",
    "        y_true_te.extend(labels['te'].numpy())\n",
    "\n",
    "y_true_exp, y_true_icm, y_true_te = np.array(y_true_exp), np.array(y_true_icm), np.array(y_true_te)\n",
    "y_pred_exp, y_pred_icm, y_pred_te = np.array(y_pred_exp), np.array(y_pred_icm), np.array(y_pred_te)\n",
    "\n",
    "# Exact Gardner Match Rate\n",
    "exact_match = (y_true_exp == y_pred_exp) & (y_true_icm == y_pred_icm) & (y_true_te == y_pred_te)\n",
    "exact_match_rate = np.mean(exact_match)\n",
    "print(f\"Exact Gardner Match Rate: {exact_match_rate:.4f}\")\n",
    "\n",
    "# 2. Per-head metrics\n",
    "print(f\"Expansion Accuracy: {accuracy_score(y_true_exp, y_pred_exp):.4f}\")\n",
    "print(f\"ICM Accuracy: {accuracy_score(y_true_icm, y_pred_icm):.4f}\")\n",
    "print(f\"TE Accuracy: {accuracy_score(y_true_te, y_pred_te):.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Confusion matrices for each head\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "heads = [('Expansion', y_true_exp, y_pred_exp), ('ICM', y_true_icm, y_pred_icm), ('TE', y_true_te, y_pred_te)]\n",
    "\n",
    "for i, (name, y_t, y_p) in enumerate(heads):\n",
    "    cm = confusion_matrix(y_t, y_p)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])\n",
    "    axes[i].set_title(f'{name} Confusion Matrix')\n",
    "    axes[i].set_ylabel('True Label')\n",
    "    axes[i].set_xlabel('Predicted Label')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Sample predictions & 5. Failure Analysis (Highlighting Grade C / minority)\n",
    "print(\"\\n--- Sample Predictions & Failure Analysis ---\")\n",
    "\n",
    "def map_labels(e, i, t):\n",
    "    # Assuming mapping dict exists in your scope\n",
    "    return f\"{e}{chr(i+65)}{chr(t+65)}\"\n",
    "\n",
    "plt.figure(figsize=(15, 8))\n",
    "count = 0\n",
    "for images, labels in val_loader:\n",
    "    images = images.to(device)\n",
    "    out_exp, out_icm, out_te = model(images)\n",
    "    \n",
    "    preds_e = torch.argmax(out_exp, dim=1).cpu().numpy()\n",
    "    preds_i = torch.argmax(out_icm, dim=1).cpu().numpy()\n",
    "    preds_t = torch.argmax(out_te, dim=1).cpu().numpy()\n",
    "    \n",
    "    true_e = labels['expansion'].numpy()\n",
    "    true_i = labels['icm'].numpy()\n",
    "    true_t = labels['te'].numpy()\n",
    "    \n",
    "    for idx in range(len(images)):\n",
    "        if count >= 8: break\n",
    "        \n",
    "        # Identify if this is a failure or minority class case\n",
    "        is_minority_or_fail = (true_i[idx] == 2 or true_t[idx] == 2) or (true_e[idx] != preds_e[idx] or true_i[idx] != preds_i[idx] or true_t[idx] != preds_t[idx])\n",
    "        \n",
    "        plt.subplot(2, 4, count + 1)\n",
    "        img_vis = images[idx].cpu().numpy().transpose(1, 2, 0)\n",
    "        img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])\n",
    "        plt.imshow(np.clip(img_vis, 0, 1))\n",
    "        \n",
    "        true_grade = map_labels(true_e[idx]+1, true_i[idx], true_t[idx])\n",
    "        pred_grade = map_labels(preds_e[idx]+1, preds_i[idx], preds_t[idx])\n",
    "        \n",
    "        color = 'red' if is_minority_or_fail and true_grade != pred_grade else 'green'\n",
    "        title = f\"True: {true_grade}\\nPred: {pred_grade}\"\n",
    "        if true_i[idx] == 2 or true_t[idx] == 2:\n",
    "            title += \"\\n(Minority C)\"\n",
    "            \n",
    "        plt.title(title, color=color)\n",
    "        plt.axis('off')\n",
    "        count += 1\n",
    "    if count >= 8: break\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
]
append_to_notebook('research/embryo_research_malpani.ipynb', cells_2)

# 3. Update embryo_validator.ipynb
cells_3 = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# SECTION: VALIDATION AUDIT\n",
    "\n",
    "This section verifies that the validator accurately filters out non-embryo images, displaying precision/recall metrics, and conducting individual test cases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix\n",
    "\n",
    "# 1. & 2. Metrics Calculation\n",
    "# Assuming val_generator exists\n",
    "y_true = val_generator.classes\n",
    "y_pred_probs = model.predict(val_generator, verbose=0)\n",
    "y_pred = (y_pred_probs >= 0.85).astype(int).flatten() # 0.85 is threshold from inference\n",
    "\n",
    "print(f\"Accuracy: {accuracy_score(y_true, y_pred):.4f}\")\n",
    "print(f\"Precision: {precision_score(y_true, y_pred):.4f}\")\n",
    "print(f\"Recall: {recall_score(y_true, y_pred):.4f}\")\n",
    "\n",
    "# 3. Confusion Matrix\n",
    "cm = confusion_matrix(y_true, y_pred)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Embryo', 'Embryo'], yticklabels=['Non-Embryo', 'Embryo'])\n",
    "plt.title('Validation Model Confusion Matrix (Threshold 0.85)')\n",
    "plt.ylabel('True')\n",
    "plt.xlabel('Predicted')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Specific Test Cases\n",
    "from PIL import Image\n",
    "\n",
    "def test_image(path, label):\n",
    "    if not os.path.exists(path):\n",
    "        print(f\"[Skip] File not found for test: {path}\")\n",
    "        return\n",
    "    \n",
    "    img = Image.open(path).convert('RGB').resize(IMG_SIZE)\n",
    "    img_array = np.expand_dims(np.array(img).astype('float32'), axis=0)\n",
    "    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)\n",
    "    \n",
    "    conf = model.predict(processed_img, verbose=0)[0][0]\n",
    "    is_embryo = conf >= 0.85\n",
    "    status = \"ACCEPTED\" if is_embryo else \"REJECTED\"\n",
    "    \n",
    "    print(f\"Test Case ({label}): {status} (Confidence: {conf:.4f})\")\n",
    "    plt.imshow(img)\n",
    "    plt.title(f\"{label}\\n{status} - {conf:.2%}\")\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "\n",
    "print(\"\\n--- Visual Test Cases ---\")\n",
    "# We fallback to dummy logic if we can't find files in a notebook run, but expected paths for testing:\n",
    "test_image('2b.jpeg', 'Valid Embryo Image')\n",
    "test_image('embryo_ai_logo_1776785134392.png', 'Logo / Random Photo')\n",
    "# You can add 'screenshot.jpg' or 'noisy.jpg' below to test edge cases\n"
   ]
  }
]
append_to_notebook('embryo_validator.ipynb', cells_3)
