import numpy as np
import pandas as pd

def normalisasi(array_matriks):
    hasil = array_matriks / sum(array_matriks)
    return hasil


def fcm(dataframe, jumlah_cluster = 3, w = 2, maxIter = 100, error_threshold = 0.01):
    jumlah_data, jumlah_fitur = dataframe.shape

    matriks_myu = np.absolute(np.random.randn(jumlah_data, jumlah_cluster))
    v = np.zeros((jumlah_cluster, jumlah_fitur))

    for (indeks_baris, isi_matriks) in enumerate(matriks_myu):
        matriks_myu[indeks_baris] = normalisasi(isi_matriks)

    df_myu = pd.read_excel("coba.xlsx")
    jumlah_baris_myu, jumlah_kolom_myu = df_myu.shape
    for index_baris in range(jumlah_baris_myu):
        for index_kolom in range(jumlah_kolom_myu):
            matriks_myu[index_baris][index_kolom] = df_myu.iloc[index_baris][index_kolom]

    jumlah_fungsi_objektif_sebelumnya = 0

    lebih_error = True
    iter_sekarang = 1

    while lebih_error and iter_sekarang <= maxIter:
        d_kuadrat = np.zeros((jumlah_data, jumlah_fitur))
        matriks_myu_baru = np.zeros((jumlah_data, jumlah_cluster))
        gabungan_myu_kuadrat = pow(matriks_myu, 2)

        # cari pusat cluster
        for indeks_cluster in range(jumlah_cluster):
            myu_kuadrat = []
            for indeks_data in range(jumlah_data):
                myu_kuadrat.append(
                    pow(matriks_myu[indeks_data][indeks_cluster], w))
            jumlah_myu_kuadrat = sum(myu_kuadrat)
            for indeks_fitur in range(jumlah_fitur):
                jumlah_myukuadrat_kalifitur = 0
                for indeks_data in range(jumlah_data):
                    jumlah_myukuadrat_kalifitur += myu_kuadrat[indeks_data] * \
                        dataframe.iloc[indeks_data][indeks_fitur]
                v[indeks_cluster][indeks_fitur] = jumlah_myukuadrat_kalifitur / \
                    jumlah_myu_kuadrat

        #print(v)

        # cari myu baru 
        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                for indeks_fitur in range(jumlah_fitur):
                    d_kuadrat[indeks_data][indeks_cluster] += pow(
                        dataframe.iloc[indeks_data][indeks_fitur] - v[indeks_cluster][indeks_fitur], 2)

        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                jumlah_d_kuadrat_i_bagi_d_kuadrat_j = 0
                for j in range (jumlah_cluster):
                    jumlah_d_kuadrat_i_bagi_d_kuadrat_j += pow(
                        d_kuadrat[indeks_data][indeks_cluster] / d_kuadrat[indeks_data][j], 1 / (w - 1))
                matriks_myu_baru[indeks_data][indeks_cluster] = 1 / jumlah_d_kuadrat_i_bagi_d_kuadrat_j
        
        # cari fungsi objektif
        jumlah_fungsi_objektif = 0
        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                    jumlah_fungsi_objektif += d_kuadrat[indeks_data][indeks_cluster] * gabungan_myu_kuadrat[indeks_data][indeks_cluster]
        
        #print(jumlah_fungsi_objektif)
        #print(matriks_myu_baru)
        error = abs(jumlah_fungsi_objektif - jumlah_fungsi_objektif_sebelumnya)
        print('Iterasi ke {}, Hasil Fungsi Objektif = {}, Error = {}'.format(iter_sekarang, jumlah_fungsi_objektif, error))
        
        iter_sekarang += 1
        
        lebih_error = error > error_threshold
        matriks_myu = matriks_myu_baru
        jumlah_fungsi_objektif_sebelumnya = jumlah_fungsi_objektif
        
    print("Hasil akhir matriks FCM yang akan digunakan pada perhitungan FPCM")
    print(matriks_myu)
    return(matriks_myu, v)


def fpcm(dataframe, myu, v, jumlah_cluster = 3, w = 2, eta = 2, maxIter = 100, error_threshold = 0.01):
    jumlah_data, jumlah_fitur = dataframe.shape
    jumlah_fungsi_objektif_sebelumnya = 0
    matriks_t = np.zeros((jumlah_data, jumlah_cluster))
    d_kuadrat_lama = np.zeros((jumlah_data, jumlah_fitur))
    lebih_error = True
    iter_sekarang = 1

    #cari t baru dari myu fcm dan v fcm
    for indeks_data in range(jumlah_data):
        for indeks_cluster in range(jumlah_cluster):
            for indeks_fitur in range(jumlah_fitur):
                d_kuadrat_lama[indeks_data][indeks_cluster] += pow(
                    dataframe.iloc[indeks_data][indeks_fitur] - v[indeks_cluster][indeks_fitur], 2)
            d_kuadrat_lama[indeks_data][indeks_cluster] = 1 / d_kuadrat_lama[indeks_data][indeks_cluster]

    jumlah_t_cluster = []
    for indeks_cluster in range(jumlah_cluster):
        jumlah = 0
        for indeks_data in range(jumlah_data):
            jumlah += d_kuadrat_lama[indeks_data][indeks_cluster]
        jumlah_t_cluster.append(jumlah)
        for indeks_data in range(jumlah_data):
            matriks_t[indeks_data][indeks_cluster] = pow(d_kuadrat_lama[indeks_data][indeks_cluster] / jumlah_t_cluster[indeks_cluster], 1 / (eta - 1))

    while lebih_error and iter_sekarang <= maxIter:
        d_kuadrat_miu = np.zeros((jumlah_data, jumlah_fitur))
        d_kuadrat_t = np.zeros((jumlah_data, jumlah_fitur))
        matriks_myu_baru = np.zeros((jumlah_data, jumlah_cluster))
        matriks_t_baru = np.zeros((jumlah_data, jumlah_cluster))
        gabungan_myu_kuadrat = pow(myu, 2)
        gabungan_t_kuadrat = pow(matriks_t, 2)

        #cari pusat cluster (v)
        for indeks_cluster in range(jumlah_cluster):
            myu_kuadrat = []
            t_kuadrat = []
            for indeks_data in range(jumlah_data):
                myu_kuadrat.append(
                    pow(myu[indeks_data][indeks_cluster], w))
                t_kuadrat.append(
                    pow(matriks_t[indeks_data][indeks_cluster], eta)
                )
            jumlah_myu_t_kuadrat = sum(myu_kuadrat) + sum(t_kuadrat)
            for indeks_fitur in range(jumlah_fitur):
                jumlah_myukuadrat_kalifitur = 0
                for indeks_data in range(jumlah_data):
                    jumlah_myukuadrat_kalifitur += (myu_kuadrat[indeks_data] + t_kuadrat[indeks_data]) * \
                        dataframe.iloc[indeks_data][indeks_fitur]
                v[indeks_cluster][indeks_fitur] = jumlah_myukuadrat_kalifitur / \
                    jumlah_myu_t_kuadrat
        # print(v)

        #cari miu baru
        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                for indeks_fitur in range(jumlah_fitur):
                    d_kuadrat_miu[indeks_data][indeks_cluster] += pow(
                        dataframe.iloc[indeks_data][indeks_fitur] - v[indeks_cluster][indeks_fitur], 2)

        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                jumlah_d_kuadrat_i_bagi_d_kuadrat_j = 0
                for j in range (jumlah_cluster):
                    jumlah_d_kuadrat_i_bagi_d_kuadrat_j += pow(
                        d_kuadrat_miu[indeks_data][indeks_cluster] / d_kuadrat_miu[indeks_data][j], 1 / (w - 1))
                matriks_myu_baru[indeks_data][indeks_cluster] = 1 / jumlah_d_kuadrat_i_bagi_d_kuadrat_j

        #cari t baru
        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                for indeks_fitur in range(jumlah_fitur):
                    d_kuadrat_t[indeks_data][indeks_cluster] += pow(
                        dataframe.iloc[indeks_data][indeks_fitur] - v[indeks_cluster][indeks_fitur], 2)
                d_kuadrat_t[indeks_data][indeks_cluster] = 1 / d_kuadrat_t[indeks_data][indeks_cluster]

        jumlah_t_cluster = []
        for indeks_cluster in range(jumlah_cluster):
            jumlah = 0
            for indeks_data in range(jumlah_data):
                jumlah += d_kuadrat_t[indeks_data][indeks_cluster]
            jumlah_t_cluster.append(jumlah)
            for indeks_data in range(jumlah_data):
                matriks_t_baru[indeks_data][indeks_cluster] = pow(d_kuadrat_t[indeks_data][indeks_cluster] / jumlah_t_cluster[indeks_cluster], 1 / (eta - 1))

        # cari fungsi objektif
        jumlah_fungsi_objektif = 0
        for indeks_data in range(jumlah_data):
            for indeks_cluster in range(jumlah_cluster):
                jumlah_fungsi_objektif += d_kuadrat_miu[indeks_data][indeks_cluster] *(gabungan_myu_kuadrat[indeks_data][indeks_cluster] + gabungan_t_kuadrat[indeks_data][indeks_cluster])
        error = abs(jumlah_fungsi_objektif - jumlah_fungsi_objektif_sebelumnya)
        print('Iterasi ke {}, Hasil Fungsi Objektif = {}, Error = {}'.format(iter_sekarang, jumlah_fungsi_objektif, error))
        
        iter_sekarang += 1
        # lebih_error = abs(jumlah_fungsi_objektif - jumlah_fungsi_objektif_sebelumnya)
        lebih_error = error > error_threshold
        myu = matriks_myu_baru
        matriks_t = matriks_t_baru
        jumlah_fungsi_objektif_sebelumnya = jumlah_fungsi_objektif


    print(myu)
    print("Hasil Klusterisasi \n c1 c2 c3")
    for matriks_baris in myu:
        hasil_maksimal = max(matriks_baris)
        for x in matriks_baris:
            print(" v  " if x == hasil_maksimal else "   ", end="")
        print()

path = "logfuz.xlsx"
df = pd.read_excel(path)
(myu_fcm, v_fcm) = fcm(df)
fpcm(df, myu_fcm, v_fcm)