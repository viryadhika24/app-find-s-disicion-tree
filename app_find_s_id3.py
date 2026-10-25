import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Find-S & ID3", page_icon="🖧", layout="wide")

# Dataset contoh dari kode pembelajaran
data = [
    ["Pendek", "Besar", "Sedang", "Rendah", "Gangguan"],
    ["Pendek", "Kecil", "Rendah", "Tinggi", "Gangguan"],
    ["Panjang", "Besar", "Sedang", "Tinggi", "Normal"],
    ["Panjang", "Kecil", "Tinggi", "Rendah", "Normal"],
    ["Pendek", "Besar", "Tinggi", "Tinggi", "Gangguan"],
    ["Panjang", "Kecil", "Rendah", "Tinggi", "Gangguan"],
    ["Panjang", "Kecil", "Tinggi", "Rendah", "Gangguan"],
    ["Panjang", "Kecil", "Sedang", "Rendah", "Normal"],
    ["Panjang", "Besar", "Tinggi", "Tinggi", "Normal"],
    ["Panjang", "Kecil", "Sedang", "Rendah", "Gangguan"],
    ["Pendek", "Besar", "Sedang", "Tinggi", "Normal"],
    ["Panjang", "Besar", "Rendah", "Tinggi", "Normal"],
]
cols = ["Waktu", "Paket", "Frekuensi", "Prioritas", "Gangguan"]
df = pd.DataFrame(data, columns=cols)
features, target = cols[:-1], cols[-1]

def find_s(dataframe, target_col, positive):
    attrs = [c for c in dataframe.columns if c != target_col]
    h = ["Ø"] * len(attrs)
    history = [{"Baris": "-", "Kelas": "-", "Hipotesis": "⟨" + ", ".join(h) + "⟩",
                "Keterangan": "Hipotesis awal"}]
    for i, (_, row) in enumerate(dataframe.iterrows(), start=1):
        if row[target_col] == positive:
            for j, attr in enumerate(attrs):
                if h[j] == "Ø":
                    h[j] = row[attr]
                elif h[j] != row[attr]:
                    h[j] = "?"
            note = "Hipotesis diperbarui dari contoh positif"
        else:
            note = "Diabaikan: bukan contoh positif"
        history.append({"Baris": i, "Kelas": row[target_col],
                        "Hipotesis": "⟨" + ", ".join(h) + "⟩", "Keterangan": note})
    return attrs, h, pd.DataFrame(history)

def entropy(series):
    if len(series) == 0:
        return 0.0
    p = series.value_counts(normalize=True)
    return -sum(float(x) * math.log2(float(x)) for x in p if x > 0)

def information_gain(dataframe, attr, target_col):
    parent = entropy(dataframe[target_col])
    child = 0.0
    for value in dataframe[attr].unique():
        subset = dataframe[dataframe[attr] == value]
        child += len(subset) / len(dataframe) * entropy(subset[target_col])
    return parent - child

def id3(dataframe, attrs, target_col):
    if dataframe[target_col].nunique() == 1:
        return dataframe[target_col].iloc[0]
    if not attrs:
        return dataframe[target_col].value_counts().idxmax()
    gains = {a: information_gain(dataframe, a, target_col) for a in attrs}
    best = max(gains, key=gains.get)
    tree = {best: {}}
    remaining = [a for a in attrs if a != best]
    for value in dataframe[best].unique():
        subset = dataframe[dataframe[best] == value]
        tree[best][value] = id3(subset, remaining, target_col)
    return tree

def predict(tree, row):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = row[attr]
    branches = tree[attr]
    if value not in branches:
        return "Tidak diketahui"
    return predict(branches[value], row)

def tree_lines(tree, depth=0):
    lines = []
    indent = "    " * depth
    if not isinstance(tree, dict):
        return [indent + "→ " + str(tree)]
    for attr, branches in tree.items():
        lines.append(indent + attr)
        for value, child in branches.items():
            lines.append(indent + f"├── {value}")
            lines.extend(tree_lines(child, depth + 1))
    return lines

def rules_from_tree(tree, conditions=None):
    conditions = [] if conditions is None else conditions
    if not isinstance(tree, dict):
        return ["IF " + (" AND ".join(conditions) if conditions else "TRUE")
                + " THEN Gangguan = " + str(tree)]
    result = []
    for attr, branches in tree.items():
        for value, child in branches.items():
            result.extend(rules_from_tree(child, conditions + [f"{attr} = {value}"]))
    return result

attrs, hypothesis, history = find_s(df, target, "Gangguan")
gains = {a: information_gain(df, a, target) for a in features}
tree = id3(df, features.copy(), target)
preds = [predict(tree, row) for _, row in df.iterrows()]
result = df.copy()
result["Prediksi ID3"] = preds
result["Sesuai?"] = result[target] == result["Prediksi ID3"]
correct = int(result["Sesuai?"].sum())
accuracy = correct / len(df) * 100

st.title("Pembelajaran Find-S & Decision Tree ID3")
st.caption("Studi kasus pembelajaran: deteksi gangguan jaringan komputer")
st.info("Ini adalah aplikasi pembelajaran dengan 12 data contoh, bukan alat diagnosis jaringan nyata.")

t1, t2, t3, t4, t5 = st.tabs(["Dataset", "Find-S", "Decision Tree ID3", "Coba Prediksi", "Catatan"])

with t1:
    a, b, c = st.columns(3)
    a.metric("Jumlah data", len(df))
    b.metric("Label Gangguan", int((df[target] == "Gangguan").sum()))
    c.metric("Label Normal", int((df[target] == "Normal").sum()))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Unduh dataset CSV", df.to_csv(index=False).encode("utf-8"),
                       "dataset_gangguan_jaringan.csv", "text/csv")

with t2:
    st.subheader("Simulasi Find-S")
    st.write("Kelas positif yang dipilih: **Gangguan**")
    st.write("Hipotesis akhir: **⟨" + ", ".join(hypothesis) + "⟩**")
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.markdown("`Ø` = atribut belum ditetapkan; `?` = nilai atribut digeneralisasi.")
    st.warning(
        "Hasil: akurasi 50% dihitung dengan asumsi semua data diprediksi Gangguan "
        "karena hipotesis akhir seluruhnya ?. Itu bukan metrik akurasi bawaan Find-S; "
        "tampilkan sebagai ilustrasi asumsi saja, bukan evaluasi model yang setara dengan ID3."
    )

with t3:
    st.subheader("Perhitungan ID3")
    st.metric("Entropy awal", f"{entropy(df[target]):.5f}")
    gain_df = pd.DataFrame({"Atribut": list(gains.keys()),
                            "Information Gain": [round(x, 5) for x in gains.values()]})
    st.dataframe(gain_df.sort_values("Information Gain", ascending=False),
                 use_container_width=True, hide_index=True)
    st.write("**Root:**", max(gains, key=gains.get))
    st.markdown("#### Struktur pohon")
    st.code("\n".join(tree_lines(tree)), language="text")
    st.markdown("#### Aturan IF–THEN")
    for i, rule in enumerate(rules_from_tree(tree), 1):
        st.write(f"{i}. `{rule}`")
    st.markdown("#### Prediksi pada data pembentuk pohon")
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.metric("Kecocokan pada data pembentuk pohon", f"{accuracy:.2f}%")
    st.warning("Ini hasil pada data yang digunakan membangun pohon (data latih), bukan bukti performa pada data baru.")

with t4:
    st.subheader("Coba prediksi satu kombinasi atribut")
    with st.form("prediction"):
        c1, c2 = st.columns(2)
        with c1:
            waktu = st.selectbox("Waktu", sorted(df["Waktu"].unique()))
            paket = st.selectbox("Paket", sorted(df["Paket"].unique()))
        with c2:
            frekuensi = st.selectbox("Frekuensi", sorted(df["Frekuensi"].unique()))
            prioritas = st.selectbox("Prioritas", sorted(df["Prioritas"].unique()))
        submitted = st.form_submit_button("Prediksi")
    if submitted:
        sample = {"Waktu": waktu, "Paket": paket, "Frekuensi": frekuensi, "Prioritas": prioritas}
        outcome = predict(tree, sample)
        st.write("Input:", sample)
        if outcome == "Gangguan":
            st.error("Prediksi pohon ID3: Gangguan")
        elif outcome == "Normal":
            st.success("Prediksi pohon ID3: Normal")
        else:
            st.warning("Kombinasi ini tidak memiliki cabang yang sesuai pada pohon.")
        st.caption("Prediksi mengikuti pola dataset contoh; bukan kepastian kondisi jaringan nyata.")

with t5:
    st.subheader("Catatan untuk presentasi")
    st.markdown("""
**Find-S**
- Memulai dari hipotesis paling spesifik (`Ø`).
- Memperbarui hipotesis hanya menggunakan contoh positif (`Gangguan`).
- `?` berarti atribut tidak membatasi nilai tertentu.
- Find-S tidak membentuk pohon keputusan.

**Decision Tree ID3**
- Entropy mengukur ketidakmurnian label.
- Information Gain mengukur pengurangan entropy setelah data dibagi berdasarkan atribut.
- Atribut dengan Gain tertinggi dipilih sebagai node, lalu proses diulang pada cabang.

**Keterbatasan**
- Dataset hanya 12 baris dan ditujukan untuk latihan.
- Kecocokan pada data latih dapat terlihat tinggi karena model belajar dari data yang sama.
- Penggunaan nyata memerlukan data yang lebih banyak, relevan, dan pengujian pada data terpisah.
""")
