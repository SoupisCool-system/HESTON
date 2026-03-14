#Mô phỏng + dự đoán giá 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PHẦN 1: NHẬP THAM SỐ TỪ TERMINAL
# ==============================================================================
print("=" * 60)
print("   ĐỊNH GIÁ OPTION & PHÂN TÍCH RỦI RO (HESTON MODEL)")
print("=" * 60)
print()

try:
    # 1. Dữ liệu thị trường hiện tại
    print("--- 1. Dữ liệu thị trường ---")
    S0 = float(input("  Giá cổ phiếu hiện tại (vd: 412): "))
    K  = float(input("  Giá Strike (vd: 450): "))
    T_days = int(input("  Số ngày dự báo T_days (vd: 77): "))
    r  = float(input("  Lãi suất không rủi ro (vd: 0.05 cho 5%): "))
    print()

    # 2. Dữ liệu Biến động (Volatility)
    print("--- 2. Dữ liệu Volatility ---")
    IV_current = float(input("  IV hiện tại (vd: 0.58 cho 58%): "))
    IV_longterm = float(input("  IV trung bình lịch sử (vd: 0.76 cho 76%): "))
    print()

    # 3. Tham số Heston
    print("--- 3. Tham số Heston ---")
    kappa = float(input("  Kappa - tốc độ hồi quy (vd: 3.0): "))
    xi    = float(input("  Xi - Vol of Vol (vd: 0.9): "))
    rho   = float(input("  Rho - tương quan (vd: -0.7): "))
    print()

    # 4. Cấu hình mô phỏng
    print("--- 4. Cấu hình mô phỏng ---")
    barrier_level = float(input("  Barrier level / Stoploss (vd: 380): "))
    num_sims = int(input("  Số lượng mô phỏng num_sims (vd: 10000): "))

except ValueError:
    print("\n[LỖI] Giá trị nhập không hợp lệ! Sử dụng giá trị mặc định.")
    S0 = 412.0
    K  = 450.0
    T_days = 77
    r  = 0.05
    IV_current = 0.58
    IV_longterm = 0.76
    kappa = 3.0
    xi    = 0.9
    rho   = -0.7
    barrier_level = 380
    num_sims = 10000

# Hiển thị lại tham số đã nhập
print()
print("-" * 60)
print(f"  Giá cổ phiếu   : ${S0}")
print(f"  Giá Strike     : ${K}")
print(f"  T_days         : {T_days} ngày")
print(f"  Lãi suất       : {r*100:.1f}%")
print(f"  IV hiện tại    : {IV_current*100:.1f}%")
print(f"  IV lịch sử     : {IV_longterm*100:.1f}%")
print(f"  Kappa          : {kappa}")
print(f"  Xi             : {xi}")
print(f"  Rho            : {rho}")
print(f"  Barrier/SL     : ${barrier_level}")
print(f"  Num Sims       : {num_sims:,}")
print("-" * 60)
print("\nĐang chạy mô phỏng...")

# ==============================================================================
# PHẦN 2: MÔ HÌNH TÍNH TOÁN HESTON (QE & BROADIE-KAYA)
# ==============================================================================
class HestonPricingEngine_V2:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def simulate_paths(self, T, num_steps, n_paths):
        dt = T / num_steps
        exp_k = np.exp(-self.kappa * dt)
        c1 = (self.sigma**2 * exp_k / self.kappa) * (1.0 - exp_k)
        c2 = (self.theta * self.sigma**2 / (2.0 * self.kappa)) * (1.0 - exp_k)**2
        
        gamma1 = 0.5
        gamma2 = 0.5
        
        K0 = -self.rho * self.kappa * self.theta / self.sigma * dt
        K1 = gamma1 * dt * (self.kappa * self.rho / self.sigma - 0.5) - self.rho / self.sigma
        K2 = gamma2 * dt * (self.kappa * self.rho / self.sigma - 0.5) + self.rho / self.sigma
        K3 = gamma1 * dt * (1.0 - self.rho**2)
        K4 = gamma2 * dt * (1.0 - self.rho**2)
        
        V = np.full(n_paths, self.v0)
        X = np.full(n_paths, np.log(self.S0))
        
        V_paths = np.zeros((n_paths, num_steps + 1))
        X_paths = np.zeros((n_paths, num_steps + 1))
        V_paths[:, 0] = V
        X_paths[:, 0] = X
        
        for t in range(1, num_steps + 1):
            # Variance process
            m = self.theta + (V - self.theta) * exp_k
            s2 = V * c1 + c2
            psi = s2 / (m**2)
            
            V_new = np.zeros_like(V)
            
            # Quadratic scheme
            idx_quad = psi <= 1.5
            if np.any(idx_quad):
                psi_quad = psi[idx_quad]
                m_quad = m[idx_quad]
                b2 = 2.0 / psi_quad - 1.0 + np.sqrt(2.0 / psi_quad * (2.0 / psi_quad - 1.0))
                a = m_quad / (1.0 + b2)
                Z_V = np.random.standard_normal(size=np.sum(idx_quad))
                V_new[idx_quad] = a * (np.sqrt(b2) + Z_V)**2

            # Exponential scheme
            idx_exp = psi > 1.5
            if np.any(idx_exp):
                psi_exp = psi[idx_exp]
                m_exp = m[idx_exp]
                p = (psi_exp - 1.0) / (psi_exp + 1.0)
                beta = (1.0 - p) / m_exp
                U = np.random.uniform(size=np.sum(idx_exp))
                safe_ratio = np.maximum((1.0 - p) / (1.0 - U), 1e-12)
                V_new[idx_exp] = np.where(U <= p, 0.0, (1.0 / beta) * np.log(safe_ratio))
            
            # Price process (Broadie-Kaya)
            Z_X = np.random.standard_normal(size=n_paths)
            sqrt_inner = np.maximum(K3 * V + K4 * V_new, 0.0)
            X_new = X + self.r * dt + K0 + K1 * V + K2 * V_new + np.sqrt(sqrt_inner) * Z_X
            
            V = V_new
            X = X_new
            
            V_paths[:, t] = V
            X_paths[:, t] = X
            
        return np.exp(X_paths), V_paths

# Chuyển đổi tham số sang năm
T = T_days / 365.0
v0 = IV_current ** 2
theta = IV_longterm ** 2
num_steps = T_days  # Mỗi bước là 1 ngày

# Khởi tạo mô hình và chạy
engine = HestonPricingEngine_V2(S0=S0, v0=v0, kappa=kappa, theta=theta, sigma=xi, rho=rho, r=r)
St, vt = engine.simulate_paths(T=T, num_steps=num_steps, n_paths=num_sims)

# ==============================================================================
# PHẦN 3: PHÂN TÍCH KẾT QUẢ & IN BÁO CÁO
# ==============================================================================

# Lấy giá trị ngày cuối cùng
final_prices = St[:, -1]
final_vols = np.sqrt(np.maximum(vt[:, -1], 0))

# 1. Định giá CALL Option (Payoff = Max(S - K, 0))
payoffs_call = np.maximum(final_prices - K, 0)
fair_value_call = np.exp(-r * T) * np.mean(payoffs_call)

# 2. Định giá PUT Option (Payoff = Max(K - S, 0))
payoffs_put = np.maximum(K - final_prices, 0)
fair_value_put = np.exp(-r * T) * np.mean(payoffs_put)

# 3. Tính xác suất ITM (In The Money)
prob_ITM_call = np.mean(final_prices > K) * 100
prob_ITM_put  = np.mean(final_prices < K) * 100

# 4. Tính xác suất chạm Stoploss (Barrier Touch)
# Kiểm tra xem min giá trong suốt hành trình có < barrier không
min_prices = np.min(St, axis=1)
prob_touch_barrier = np.mean(min_prices < barrier_level)

# 5. Tính Value at Risk (VaR 95%)
var_95 = np.percentile(final_prices, 5)

# --- IN KẾT QUẢ RA MÀN HÌNH ---
print("\n" + "="*60)
print(f"   BÁO CÁO ĐỊNH GIÁ OPTION & RỦI RO (HESTON MODEL)")
print(f"   Cổ phiếu: ${S0} | Strike: ${K} | Thời hạn: {T_days} ngày")
print("="*60)

print(f"\n--- 1. DỰ BÁO XU HƯỚNG (Sau {T_days} ngày) ---")
print(f"• Giá trung bình dự báo:   ${np.mean(final_prices):.2f}")
print(f"• IV dự báo trung bình:    {np.mean(final_vols)*100:.1f}% (Hiện tại: {IV_current*100:.1f}%)")
print(f"  -> Xu hướng Volatility:  {'GIẢM (Mean Reversion)' if np.mean(final_vols) < IV_current else 'TĂNG'}")

print(f"\n--- 2. KẾT QUẢ ĐỊNH GIÁ (FAIR VALUE) ---")
print(f"➤ CALL OPTION (Mua lên):  ${fair_value_call:.2f}")
print(f"  • Xác suất thắng (ITM): {prob_ITM_call:.1f}%")
print(f"  • Chiến lược: Nếu giá thị trường < ${fair_value_call:.2f} -> NÊN MUA CALL")

print(f"\n➤ PUT OPTION (Bán xuống): ${fair_value_put:.2f}")
print(f"  • Xác suất thắng (ITM): {prob_ITM_put:.1f}%")
print(f"  • Chiến lược: Nếu giá thị trường > ${fair_value_put:.2f} -> NÊN BÁN PUT")

print(f"\n--- 3. QUẢN TRỊ RỦI RO (Risk Management) ---")
print(f"• Mức Stoploss giả định:   ${barrier_level}")
print(f"• Xác suất BỊ QUÉT lệnh:   {prob_touch_barrier*100:.1f}% (Rất quan trọng!)")
print(f"• Rủi ro đuôi (VaR 95%):   Có 5% khả năng giá về dưới ${var_95:.2f}")
print(f"• Giá cao nhất có thể đạt: ${np.max(final_prices):.2f}")
print(f"• Giá thấp nhất có thể đạt: ${np.min(final_prices):.2f}")


# --- VẼ BIỂU ĐỒ ---
plt.figure(figsize=(18, 7)) # Tăng kích thước figure để chứa 3 subplot

# Biểu đồ 1: Đường đi của giá (Price Paths)
plt.subplot(1, 3, 1) # 1 hàng, 3 cột, vị trí 1
plt.plot(St[:100].T, color='gray', alpha=0.3, linewidth=1) # Vẽ 100 đường mẫu
plt.plot(np.mean(St, axis=0), color='red', linewidth=2, label='Trung bình')
plt.axhline(y=barrier_level, color='blue', linestyle='--', label=f'Stoploss ({barrier_level})')
plt.title(f'Mô phỏng 100 kịch bản giá (Heston)')
plt.xlabel('Ngày')
plt.ylabel('Giá ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 2: Phân phối xác suất giá cuối cùng (Distribution)
plt.subplot(1, 3, 2) # 1 hàng, 3 cột, vị trí 2
sns.histplot(final_prices, bins=50, kde=True, color='skyblue', stat='probability')
plt.axvline(x=S0, color='red', linestyle='--', label='Giá hiện tại')
plt.axvline(x=K, color='green', linestyle=':', label=f'Strike ({K})') # Thêm đường strike
plt.axvline(x=var_95, color='orange', linestyle='--', label='VaR 95%')
plt.title(f'Phân phối giá sau {T_days} ngày')
plt.xlabel('Giá cổ phiếu')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 3: Phân phối giá cổ phiếu so với Strike (for Option Payoff visualization)
plt.subplot(1, 3, 3) # 1 hàng, 3 cột, vị trí 3
sns.histplot(final_prices, bins=50, kde=True, color='teal', alpha=0.6)
plt.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike Price ({K})')
plt.axvline(x=S0, color='black', linestyle='-', label=f'Giá Hiện Tại ({S0})')
plt.title(f'Phân phối giá cổ phiếu so với Strike {K}')
plt.xlabel('Giá cổ phiếu khi đáo hạn')
plt.ylabel('Số lượng kịch bản')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


