#Dự đoán stochastic volatility bằng Heston model
import numpy as np
import matplotlib.pyplot as plt

# --- 1. NHẬP THAM SỐ TỪ TERMINAL ---
print("=" * 50)
print("  DỰ BÁO STOCHASTIC VOLATILITY (HESTON MODEL)")
print("=" * 50)
print()

try:
    IV_current = float(input("Nhập IV hiện tại (vd: 0.72 cho 72%): "))
    IV_longterm = float(input("Nhập Long-term IV (vd: 0.60 cho 60%): "))
    kappa = float(input("Nhập kappa - tốc độ hồi quy (vd: 3.0): "))
    xi = float(input("Nhập xi - độ biến động của volatility (vd: 0.9): "))
    T_days = int(input("Nhập T_days - số ngày dự báo (vd: 7): "))
except ValueError:
    print("\n[LỖI] Giá trị nhập không hợp lệ! Sử dụng giá trị mặc định.")
    IV_current = 0.72
    IV_longterm = 0.60
    kappa = 3.0
    xi = 0.9
    T_days = 7

# Hiển thị lại tham số đã nhập
print()
print("-" * 50)
print(f"  IV hiện tại    : {IV_current*100:.1f}%")
print(f"  Long-term IV   : {IV_longterm*100:.1f}%")
print(f"  Kappa          : {kappa}")
print(f"  Xi             : {xi}")
print(f"  T_days         : {T_days} ngày")
print("-" * 50)
print()

dt = (T_days/365) / T_days
num_sims = 10000

# --- 2. TÍNH TOÁN LẠI CHỈ RIÊNG VOLATILITY ---
v0 = IV_current**2
theta = IV_longterm**2
vt = np.zeros((num_sims, T_days + 1))
vt[:, 0] = v0

# Chạy mô phỏng CIR (Cox-Ingersoll-Ross) cho Variance
for t in range(T_days):
    Z = np.random.normal(0, 1, num_sims)
    # Lưu ý: Volatility process cũng có ngẫu nhiên (Z)
    vt_prev = np.maximum(vt[:, t], 0)
    d_vt = kappa * (theta - vt_prev) * dt + xi * np.sqrt(vt_prev) * np.sqrt(dt) * Z
    vt[:, t+1] = vt[:, t] + d_vt

# Chuyển đổi từ Variance sang Volatility (%)
vol_paths = np.sqrt(np.maximum(vt, 0)) * 100

# --- 3. VẼ BIỂU ĐỒ "NÓN BIẾN ĐỘNG" ---
plt.figure(figsize=(12, 6))

# Vẽ 100 đường mẫu (Spaghetti plot)
plt.plot(vol_paths[:100].T, color='grey', alpha=0.2, linewidth=1)

# Vẽ đường trung bình (Kỳ vọng của thị trường)
mean_vol = np.mean(vol_paths, axis=0)
plt.plot(mean_vol, color='red', linewidth=3, label='Trung bình dự báo (Mean Reversion)')

# Vẽ đường dài hạn (Long-run Average)
plt.axhline(y=IV_longterm*100, color='blue', linestyle='--', label=f'Mức cân bằng dài hạn ({IV_longterm*100}%)')

plt.title(f'Dự báo Stochastic Volatility trong {T_days} ngày tới')
plt.ylabel('Volatility (%)')
plt.xlabel('Ngày')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# In kết quả
print(f"--- KẾT QUẢ DỰ BÁO VOLATILITY ---")
print(f"1. Hiện tại: {IV_current*100:.1f}%")
print(f"2. Sau {T_days} ngày (Trung bình): {mean_vol[-1]:.1f}%")
print(f"3. Nhận định: Xu hướng biến động đang {'GIẢM' if mean_vol[-1] < IV_current*100 else 'TĂNG'}")
