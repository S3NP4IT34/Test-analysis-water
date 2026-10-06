import os
import sys
import numpy as np
import pandas as pd

papochka = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# Пороги 
MAX_SUTKI = 72     # м^3 в сутки: больше физически невозможно
SKACHOK = 300      # м^3: скачок показания больше этого = сбой счётчика
VSPLESK = 5        # м^3 в сутки: абсолютный порог всплеска
RAZ = 10           # всплеск: во сколько раз больше обычного расхода прибора
UTECHKA = 60       # суток подряд без нулевых = утечка
NOL = 30           # дней без изменения показаний = нулевое потребление
STUPENKA = 3       # ступенька: во сколько раз 2-я половина отличается от 1-й

#  Читаем файлы 
def read(name):
    return pd.read_csv(os.path.join(papochka, name), sep=";", encoding="utf-8-sig")

p = read("pokazaniya.csv")
p["data"] = pd.to_datetime(p["data"])
p = p.sort_values(["anon_id", "data"])
sobytiya = read("sobytiya.csv")
sobytiya["data"] = pd.to_datetime(sobytiya["data"])
pribory = read("pribory.csv")

#  Считаем суточный расход 
g = p.groupby("anon_id")
p["prirost"] = g["pokazanie"].diff()               # показание минус вчерашнее
p["dney"] = g["data"].diff().dt.days               # сколько дней прошло между записями
p["prev_data"] = g["data"].shift()                 # дата предыдущей записи
p["rashod"] = p["prirost"].where(p["dney"] == 1)   # суточный расход - только если записи в соседние дни

# нереальное: огромный скачок или расход больше 72 м3 в сутки
p["nereal"] = (p["prirost"].abs() > SKACHOK) | (p["rashod"] > MAX_SUTKI)
# "чистый" расход: от 0 до 72 и без сбоев (на нём считаем всплески, утечки, ступеньки)
p["chisty"] = p["rashod"].where((p["rashod"] >= 0) & (p["rashod"] <= MAX_SUTKI) & ~p["nereal"])

def ids(maska):
    """Какие приборы попали под условие."""
    return sorted(p.loc[maska, "anon_id"].unique())

def top(series, n=5, ascending=False):
    """Первые n приборов из Series (в индексе anon_id)."""
    return ", ".join(str(i) for i in series.sort_values(ascending=ascending).head(n).index)

otvety = []    # сюда складываем ответы на вопросы

# 1) Отрицательный расход 
neg = p["prirost"] < -0.000001            # меньше нуля (совсем крошечный шум округления не считаем)
neg_ids = ids(neg)
sboy_neg = ids(neg & p["nereal"])         # из них - большие скачки
otvety.append([1, "Отрицательный расход", len(neg_ids), top(p[neg].groupby("anon_id")["prirost"].min(), ascending=True),
    f"Показание уменьшилось у {len(neg_ids)} приборов ({neg.sum()} раз). Из них {len(sboy_neg)} - большие скачки (сбой счётчика), "
    "остальные - небольшие падения. Это значит: счётчик назад не идёт, значит сбой электроники или передачи, сброс или замена прибора."])

# 2) Всплески 
mediana = p[p["chisty"] > 0.001].groupby("anon_id")["chisty"].median()     # обычный расход прибора
p["mediana"] = p["anon_id"].map(mediana)
vspl_abs = p["chisty"] > VSPLESK                                           # абсолютный порог
vspl_rel = (p["chisty"] > RAZ * p["mediana"]) & (p["chisty"] > 1)          # относительный порог
vspl_ids = ids(vspl_abs | vspl_rel)
maks = p[(vspl_abs | vspl_rel) & ~p["nereal"]].groupby("anon_id")["chisty"].max()
otvety.append([2, "Всплески", len(vspl_ids), top(maks),
    f"Больше {VSPLESK} м3/сут: {len(ids(vspl_abs))} приборов. Больше {RAZ} раз обычного расхода прибора: {len(ids(vspl_rel))} приборов. "
    f"Всего {len(vspl_ids)} приборов."])

# 3) Нереальные значения 
nereal_ids = ids(p["nereal"])
sboi = p[p["nereal"]].groupby("anon_id")["prirost"].apply(lambda x: x.abs().max())
otvety.append([3, "Нереальные значения", len(nereal_ids), top(sboi),
    f"{len(nereal_ids)} приборов: скачок показания больше {SKACHOK} м3 или расход больше {MAX_SUTKI} м3 в сутки. "
    "У части скачки в миллионы м3. Показаниям этих приборов доверять нельзя."])

# 4) Утечка 
# для каждого прибора считаем самую длинную серию суток подряд с расходом больше нуля
utechka = {}
for aid, gr in p.groupby("anon_id"):
    seriya = 0
    luchshaya = 0
    for x in gr["chisty"]:
        if x > 0.001:
            seriya += 1
            luchshaya = max(luchshaya, seriya)
        else:
            seriya = 0                   # нулевой день - серия оборвалась
    utechka[aid] = luchshaya
utechka = pd.Series(utechka)
utechka_ids = utechka[utechka >= UTECHKA]
otvety.append([4, "Утечка", len(utechka_ids), top(utechka_ids),
    f"{len(utechka_ids)} приборов не имели ни одних нулевых суток {UTECHKA} дней подряд и дольше. "
    f"Для сравнения: 90+ дней - {(utechka >= 90).sum()}, 180+ дней - {(utechka >= 180).sum()}."])

# 5) Нулевое потребление при живых пакетах 
# ищем самый длинный период, где показание не менялось (при этом записи приходили)
nol = {}
for aid, gr in p.groupby("anon_id"):
    luchshee = 0
    start = None
    zapisey = 0
    for data, prirost, prev_data in zip(gr["data"], gr["prirost"], gr["prev_data"]):
        if abs(prirost) <= 0.001:         # показание не изменилось
            if start is None:
                start = prev_data         # период начался с прошлой записи
                zapisey = 0
            zapisey += 1
            if zapisey >= 10:             # не меньше 10 записей: прибор на связи
                luchshee = max(luchshee, (data - start).days)
        else:
            start = None
    nol[aid] = luchshee
nol = pd.Series(nol)
nol_ids = nol[nol >= NOL]
bs = pribory.set_index("anon_id")["bs"]
bs04 = (nol_ids.index.map(bs) == "BS-04").sum()
otvety.append([5, "Нулевое потребление", len(nol_ids), top(nol_ids[nol_ids.index.map(bs) != "BS-04"]),
    f"{len(nol_ids)} приборов не меняли показание {NOL} дней и дольше при этом присылая данные. "
    f"Из них {bs04} - станция BS-04, там почти у всех приборов показание около нуля: это проблема станции или партии приборов, "
    "а не пустых квартир. Примеры - без BS-04."])

# 6) Ступенька 
seredina = p["data"].min() + (p["data"].max() - p["data"].min()) / 2       # середина периода
a = p[p["data"] < seredina].groupby("anon_id")["chisty"].agg(["mean", "count"])
b = p[p["data"] >= seredina].groupby("anon_id")["chisty"].agg(["mean", "count"])
st = a.join(b, lsuffix="1", rsuffix="2").dropna()
st = st[(st["count1"] >= 60) & (st["count2"] >= 60)]                       # достаточно данных в обеих половинах
st["vo_skolko"] = (st["mean2"] + 0.001) / (st["mean1"] + 0.001)
vverh = st[(st["vo_skolko"] >= STUPENKA) & (st["mean2"] - st["mean1"] >= 0.05)]
vniz = st[(st["vo_skolko"] <= 1 / STUPENKA) & (st["mean1"] - st["mean2"] >= 0.05)]
vverh_chist = vverh[~vverh.index.isin(nereal_ids)]       # для примеров берём приборы без сбоев
vniz_chist = vniz[~vniz.index.isin(nereal_ids)]
otvety.append([6, "Ступенька", len(vverh) + len(vniz), "вверх: " + top(vverh_chist["mean2"], 3) + "; вниз: " + top(vniz_chist["mean1"], 3),
    f"Средний расход во 2-й половине периода в {STUPENKA}+ раза больше, чем в 1-й: {len(vverh)} приборов; в {STUPENKA}+ раза меньше: {len(vniz)} приборов. "
    "Рост - возможная утечка или новый потребитель, падение - остановка счётчика или уход потребителя."])

# 7) Связь с магнитом 
magnit = set(sobytiya.loc[sobytiya["sobytie"] == "магнит", "anon_id"])
spiski = {"отрицательный расход": set(neg_ids), "нереальные значения": set(nereal_ids), "всплески": set(vspl_ids),
          "утечка": set(utechka_ids.index), "нулевое потребление": set(nol_ids.index),
          "ступенька вверх": set(vverh.index), "ступенька вниз": set(vniz.index)}
vsego = p["anon_id"].nunique()
tablica7 = []
for nazvanie, mnozhestvo in spiski.items():
    s_magnitom = len(mnozhestvo & magnit) / len(magnit) * 100
    bez_magnita = len(mnozhestvo - magnit) / (vsego - len(magnit)) * 100
    tablica7.append([nazvanie, round(s_magnitom, 1), round(bez_magnita, 1)])
tablica7 = pd.DataFrame(tablica7, columns=["аномалия", "% приборов с магнитом", "% приборов без магнита"])
# расход за 14 дней до и после первого магнита
po_priboram = {aid: gr.set_index("data")["chisty"] for aid, gr in p.groupby("anon_id")}
pervy_magnit = sobytiya[sobytiya["sobytie"] == "магнит"].groupby("anon_id")["data"].min()
otnosheniya = []
for aid, d in pervy_magnit.items():
    ryad = po_priboram[aid]
    do = ryad[(ryad.index >= d - pd.Timedelta(days=14)) & (ryad.index < d)].mean()
    posle = ryad[(ryad.index > d) & (ryad.index <= d + pd.Timedelta(days=14))].mean()
    if do > 0.001 and pd.notna(posle):
        otnosheniya.append(posle / do)
mediana_mag = np.median(otnosheniya)
if abs(mediana_mag - 1) < 0.15:
    vyvod = "Расход после магнита почти не меняется, прямой связи нет."
else:
    vyvod = "Расход после магнита заметно меняется, связь стоит проверить."
otvety.append([7, "Связь с магнитом", len(magnit), "см. лист '7 Магнит'",
    f"Магнит был у {len(magnit)} приборов. Расход после первого магнита / до него (медиана) = {mediana_mag:.2f}. {vyvod} "
    "Приборы с магнитом не чаще остальных попадают в аномалии (см. таблицу)."])

# 8) Собственные находки 
pribory["zapisey"] = pribory["anon_id"].map(p.groupby("anon_id").size())
pribory["posledneye"] = pribory["anon_id"].map(p.groupby("anon_id")["pokazanie"].last())
pribory["sboy"] = pribory["anon_id"].isin(nereal_ids)
po_bs = pribory.groupby("bs").agg(priborov=("anon_id", "size"), mediana_zapisey=("zapisey", "median"),
                                  mediana_pokazaniya=("posledneye", "median")).round(2).reset_index()
po_tipu = pribory.groupby("tip").agg(priborov=("anon_id", "size"), so_sboem=("sboy", "sum")).reset_index()
malo = ((pribory["zapisey"] <= 5) & (pribory["bs"] == "BS-02")).sum()
otvety.append([8, "Находки", "-", "BS-02, BS-04, тип AQUA2",
    f"1) Станция BS-02 почти не присылала данные: у {malo} из 300 её приборов не больше 5 записей за год. "
    f"2) На станции BS-04 показание около нуля у почти всех приборов (медиана {po_bs.loc[po_bs['bs'] == 'BS-04', 'mediana_pokazaniya'].iloc[0]} м3). "
    f"3) Сбои счётчика почти только у старых приборов типа AQUA2 ({po_tipu.loc[po_tipu['tip'] == 'AQUA2', 'so_sboem'].iloc[0]} приборов)."])

#  Таблица по приборам: что проверять первым 
itog = pribory[["anon_id", "bs", "tip", "model", "mesyac_vypuska"]].set_index("anon_id")
itog["нереальные значения"] = itog.index.isin(nereal_ids)
itog["отрицательный расход"] = itog.index.isin(neg_ids)
itog["всплеск"] = itog.index.isin(vspl_ids)
itog["суток подряд с расходом"] = utechka
itog["дней без изменений"] = nol
itog["ступенька"] = ""
itog.loc[itog.index.isin(vverh.index), "ступенька"] = "вверх"
itog.loc[itog.index.isin(vniz.index), "ступенька"] = "вниз"
itog["магнит"] = itog.index.isin(magnit)
itog["аномалий"] = (itog["нереальные значения"].astype(int) + itog["отрицательный расход"].astype(int) + itog["всплеск"].astype(int)
                    + (itog["суток подряд с расходом"] >= UTECHKA).astype(int) + (itog["дней без изменений"] >= NOL).astype(int)
                    + (itog["ступенька"] != "").astype(int))
itog = itog.sort_values(["нереальные значения", "аномалий"], ascending=False).reset_index()
otvety.append(["итог", "Что проверять первым", "-", ", ".join(str(i) for i in itog["anon_id"].head(10)),
    "Сначала приборы со сбоем счётчика (нереальные значения) - их нужно менять или поверять, потом утечки и ступеньки. "
    "Отдельно - проблемы станций BS-02 и BS-04."])

#  Записываем Excel 
listy = {
    "Ответы": pd.DataFrame(otvety, columns=["№", "Вопрос", "Приборов", "Примеры (anon_id)", "Ответ"]).astype(str),
    "1 Отрицательный": p.loc[neg, ["anon_id", "data", "pokazanie", "prirost"]].sort_values("prirost"),
    "2 Всплески": p.loc[vspl_abs | vspl_rel, ["anon_id", "data", "chisty", "mediana"]].sort_values("chisty", ascending=False),
    "3 Нереальные": p.loc[p["nereal"], ["anon_id", "data", "pokazanie", "prirost", "dney"]],
    "4 Утечка": utechka_ids.rename("суток подряд").sort_values(ascending=False).reset_index().rename(columns={"index": "anon_id"}),
    "5 Нулевое": nol_ids.rename("дней без изменений").sort_values(ascending=False).reset_index().rename(columns={"index": "anon_id"}),
    "6 Ступенька": pd.concat([vverh.assign(направление="вверх"), vniz.assign(направление="вниз")]).reset_index(),
    "7 Магнит": tablica7,
    "8 Станции": po_bs,
    "8 Типы приборов": po_tipu,
    "Приборы": itog,
}
excel = os.path.join(papochka, "rezultaty_analiza.xlsx")
with pd.ExcelWriter(excel, engine="openpyxl") as writer:
    for nazvanie, tablica in listy.items():
        if "data" in tablica.columns:
            tablica = tablica.assign(data=tablica["data"].dt.date)      # дата без времени
        tablica.to_excel(writer, sheet_name=nazvanie, index=False)
        for kolonka in writer.sheets[nazvanie].columns:
            writer.sheets[nazvanie].column_dimensions[kolonka[0].column_letter].width = 22
    writer.sheets["Ответы"].column_dimensions["E"].width = 150
print("Excel сохранён:", excel)

#  Ответы в консоль 
print("\nОТВЕТЫ НА ВОПРОСЫ")
for nomer, vopros, kolichestvo, primery, otvet in otvety:
    print(f"\n{nomer}. {vopros}\n   Примеры: {primery}\n   Ответ: {otvet}")