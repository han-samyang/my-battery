import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks

# --- 헬퍼 및 분석 함수 (제공된 코드와 동일) ---
def find_intersection(line1_params, line2_params):
    m1, b1 = line1_params; m2, b2 = line2_params
    if abs(m1 - m2) < 1e-6: return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def analyze_lsv_methods(all_data, params):
    results = {}
    for sheet_name, data_dict in all_data.items():
        x, y = np.array(data_dict['x']), np.array(data_dict['y'])
        res = {'type': data_dict['type'], 'ox_on_v_deriv': None, 'ox_on_i_deriv': None,
               'ox_on_v_tangent': None, 'ox_on_i_tangent': None, 'tangent_lines': None}

        if len(y) < max(20, params['sw_win']):
            results[sheet_name] = res; continue
        
        high_voltage_indices = np.where(x >= params['p_start'])[0]
        if len(high_voltage_indices) > params['sw_win']:
            x_high, y_high = x[high_voltage_indices], y[high_voltage_indices]
            y_smooth = savgol_filter(y_high, window_length=params['sw_win'], polyorder=params['sw_poly'])
            grad_y, grad_x = np.gradient(y_smooth), np.gradient(x_high)
            grad_x[grad_x == 0] = 1e-9
            dI_dV = grad_y / grad_x
            
            peak_idx = None
            is_ref = 'ref' in sheet_name.lower() or 'base' in sheet_name.lower()
            
            if is_ref:
                min_peak_height = dI_dV.max() * params['peak_sensitivity']
                peaks, _ = find_peaks(dI_dV, height=min_peak_height)
                if len(peaks) > 0:
                    peak_idx = peaks[0]
            else:
                peak_idx = np.argmax(dI_dV)

            if peak_idx is not None:
                res['ox_on_v_deriv'] = x_high[peak_idx]
                res['ox_on_i_deriv'] = y_high[peak_idx]

                base_indices = np.where((x >= params['tangent_base_start']) & (x <= params['tangent_base_end']))[0]
                base_fit_params = np.polyfit(x[base_indices], y[base_indices], 1) if len(base_indices) > 1 else None
                
                ox_region_center_idx_in_high = peak_idx
                ox_region_center_idx_global = high_voltage_indices[ox_region_center_idx_in_high]
                
                start_idx = max(0, ox_region_center_idx_global - params['tangent_ox_win'] // 2)
                end_idx = min(len(x) - 1, ox_region_center_idx_global + params['tangent_ox_win'] // 2)
                ox_indices = np.arange(start_idx, end_idx + 1)
                ox_fit_params = np.polyfit(x[ox_indices], y[ox_indices], 1) if len(ox_indices) > 1 else None
                
                if base_fit_params is not None and ox_fit_params is not None:
                    v_t, i_t = find_intersection(base_fit_params, ox_fit_params)
                    if v_t and params['p_start'] < v_t < x[-1]:
                        res['ox_on_v_tangent'], res['ox_on_i_tangent'] = v_t, i_t
                        res['tangent_lines'] = {'base_params': base_fit_params, 'ox_params': ox_fit_params}
        results[sheet_name] = res
    return results

def generate_expert_commentary(summary, ref_name, add_name):
    primary_ref_data = summary.get(ref_name)
    add_data = summary.get(add_name)
    if not primary_ref_data or not add_data: return ""

    full_commentary = f"\n\n### 📝 비교 분석: {ref_name} vs. {add_name}\n\n"
    table_header = "| 분석 항목 | 기준 값 | 첨가제 값 | 성능 변화 |\n|:---|:---:|:---:|:---:|\n"
    
    ref_d_val = primary_ref_data.get('ox_on_v_deriv')
    add_d_val = add_data.get('ox_on_v_deriv')
    ref_d_str = f"{ref_d_val:.3f}V" if ref_d_val is not None else "N/A"
    add_d_str = f"{add_d_val:.3f}V" if add_d_val is not None else "N/A"
    change_d = "N/A"
    if ref_d_val is not None and add_d_val is not None:
        diff = add_d_val - ref_d_val
        change_d = f"🔺 **향상** (+{diff:.3f}V)" if diff > 0.02 else (f"🔻 **저하** ({diff:.3f}V)" if diff < -0.02 else "- (유사)")
    row_d = f"| 산화 안정성 (미분법) | {ref_d_str} | {add_d_str} | {change_d} |\n"

    ref_t_val = primary_ref_data.get('ox_on_v_tangent')
    add_t_val = add_data.get('ox_on_v_tangent')
    ref_t_str = f"{ref_t_val:.3f}V" if ref_t_val is not None else "N/A"
    add_t_str = f"{add_t_val:.3f}V" if add_t_val is not None else "N/A"
    change_t = "N/A"
    if ref_t_val is not None and add_t_val is not None:
        diff = add_t_val - ref_t_val
        change_t = f"🔺 **향상** (+{diff:.3f}V)" if diff > 0.02 else (f"🔻 **저하** ({diff:.3f}V)" if diff < -0.02 else "- (유사)")
    row_t = f"| 산화 안정성 (접선법) | {ref_t_str} | {add_t_str} | {change_t} |\n"

    full_commentary += table_header + row_d + row_t + "\n**종합 의견:**\n"
    is_d_improved, is_t_improved = '향상' in change_d, '향상' in change_t
    if is_d_improved and is_t_improved: full_commentary += "  - **(매우 긍정적)** 두 분석 방법 모두에서 일관되게 산화 안정성 향상이 관찰되어, **매우 유망한 후보**로 판단됩니다."
    elif is_d_improved or is_t_improved: full_commentary += "  - **(긍정적)** 하나의 분석 방법에서 개선 효과가 나타났습니다. 추가적인 재현성 확인이 권장됩니다."
    else: full_commentary += "  - **(개선 필요)** 기준 전해액 대비 뚜렷한 산화 안정성 개선 효과를 확인하기 어렵습니다."
            
    return full_commentary.strip()

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="배터리 LSV 분석기", layout="wide")
st.title("🔋 배터리 산화 안정성 최종 분석기 (v7.14)") # 버전 업데이트

with st.sidebar:
    st.header("⚙️ 분석 고급 설정")
    p_start = st.number_input("산화 분석 시작 전위 (V)", 2.0, 5.0, 3.5, 0.1)
    st.subheader("방법 2: 미분법 설정")
    sw_win = st.slider("평활화 윈도우", 5, 101, 21, 2)
    sw_poly = st.slider("평활화 다항식 차수", 1, 5, 3, 1)
    peak_sens = st.slider("Ref. 피크 민감도 (%)", 1, 50, 10, 1, help="Ref 샘플의 첫 피크를 찾을 때, 노이즈를 무시하기 위한 민감도입니다. (최대 기울기 대비 %)")

    st.subheader("방법 1: 접선법 설정")
    tbs = st.number_input("기준선 시작 전위", 2.0, 5.0, 3.0, 0.1)
    tbe = st.number_input("기준선 종료 전위", 2.0, 5.0, 3.5, 0.1)
    tow = st.slider("산화영역 피팅 포인트 수", 3, 51, 11, 2)

uploaded_file = st.file_uploader("분석할 엑셀 파일을 업로드하세요 (.xlsx, .xls)", type=["xlsx", "xls"])

if uploaded_file:
    all_data = {}
    try:
        xls = pd.ExcelFile(uploaded_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None, usecols=[0, 1]).apply(pd.to_numeric, errors='coerce').dropna()
            if not df.empty:
                data_type = 'Reference' if 'ref' in sheet_name.lower() or 'base' in sheet_name.lower() else 'Additive'
                all_data[sheet_name] = {'x': df.iloc[:, 0].tolist(), 'y': (df.iloc[:, 1] * 1000).tolist(), 'type': data_type}
    except Exception as e: st.error(f"엑셀 파일 로딩 오류: {e}"); st.stop()

    if not all_data: st.error("유효한 데이터 시트를 찾을 수 없습니다."); st.stop()

    ref_sheets = [name for name, data in all_data.items() if data['type'] == 'Reference']
    additive_sheets = [name for name, data in all_data.items() if data['type'] == 'Additive']
    st.success(f"데이터 처리 완료! Ref {len(ref_sheets)}개, Additive {len(additive_sheets)}개 시트를 분석합니다.")
    
    analysis_params = {'p_start': p_start, 'sw_win': sw_win, 'sw_poly': sw_poly, 
                       'peak_sensitivity': peak_sens / 100.0,
                       'tangent_base_start': tbs, 'tangent_base_end': tbe, 'tangent_ox_win': tow}
    analysis_summary = analyze_lsv_methods(all_data, analysis_params)
    
    st.subheader("📊 인터랙티브 데이터 시각화")
    
    all_sheet_names = list(all_data.keys())
    default_selection = []
    if ref_sheets: default_selection.append(ref_sheets[0])
    if additive_sheets: default_selection.append(additive_sheets[0])
    if not default_selection and all_sheet_names: default_selection = all_sheet_names[:2]

    selected_sheets = st.multiselect(
        "그래프에 표시할 데이터를 선택하세요 (다중 선택 가능):",
        options=all_sheet_names,
        default=default_selection
    )
    
    fig = go.Figure()
    
    if not selected_sheets:
        st.warning("그래프에 표시할 데이터를 위에서 1개 이상 선택해주세요.")
    else:
        for name in selected_sheets:
            data, summary = all_data.get(name), analysis_summary.get(name)
            if not data or not summary: continue
            fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='lines', name=f"{name} ({data['type'][:3]})"))
            
            if summary.get('ox_on_v_deriv'): 
                fig.add_annotation(x=summary['ox_on_v_deriv'], y=summary['ox_on_i_deriv'], 
                                   text=f"<b>미분법</b><br>{summary['ox_on_v_deriv']:.3f}V", 
                                   showarrow=True, arrowhead=2, standoff=4, ax=20, ay=30,
                                   font=dict(color="crimson", size=10), bgcolor="rgba(255,255,255,0.6)")
            if summary.get('ox_on_v_tangent'): 
                fig.add_annotation(x=summary['ox_on_v_tangent'], y=summary['ox_on_i_tangent'], 
                                   text=f"<b>접선법</b><br>{summary['ox_on_v_tangent']:.3f}V", 
                                   showarrow=True, arrowhead=2, standoff=4, ax=-20, ay=45,
                                   font=dict(color="royalblue", size=10), bgcolor="rgba(255,255,255,0.6)")

        plot_title = "데이터 비교: " + ", ".join(selected_sheets)
        fig.update_layout(title=plot_title, xaxis_title="Potential (V)", yaxis_title="Current (mA)", height=600, legend=dict(x=0.01, y=0.98))
        st.plotly_chart(fig, use_container_width=True)
    
    # --- ### 전문가 분석 코멘트 (요청사항 반영하여 수정) ### ---
    st.subheader("🔬 전문가 분석 코멘트")
    selected_refs = [s for s in selected_sheets if s in ref_sheets]
    selected_adds = [s for s in selected_sheets if s in additive_sheets]
    
    if not selected_sheets:
        pass
    elif not selected_refs or not selected_adds:
        st.info("비교 분석을 보려면 그래프 선택창에서 기준(Ref)과 첨가제(Add) 샘플을 각각 1개 이상씩 선택하세요.")
    else:
        # 1. 기준(Ref) 대비 1:1 비교 코멘트 (기존 방식 유지)
        for r_name in selected_refs:
            for a_name in selected_adds:
                commentary = generate_expert_commentary(analysis_summary, r_name, a_name)
                st.markdown(commentary, unsafe_allow_html=True)

        # 2. 첨가제 간 비교 테이블 (신규 추가)
        if len(selected_adds) > 1:
            st.markdown("---")
            st.markdown(f"#### Additive 간 비교 ({', '.join(selected_adds)})")
            
            comparison_data = []
            # 모든 첨가제 쌍에 대한 비교
            for i in range(len(selected_adds)):
                for j in range(i + 1, len(selected_adds)):
                    add1_name = selected_adds[i]
                    add2_name = selected_adds[j]
                    add1_summary = analysis_summary[add1_name]
                    add2_summary = analysis_summary[add2_name]

                    # 미분법 델타값 계산
                    diff_d = np.nan
                    if pd.notna(add1_summary['ox_on_v_deriv']) and pd.notna(add2_summary['ox_on_v_deriv']):
                        diff_d = add2_summary['ox_on_v_deriv'] - add1_summary['ox_on_v_deriv']
                    
                    # 접선법 델타값 계산
                    diff_t = np.nan
                    if pd.notna(add1_summary['ox_on_v_tangent']) and pd.notna(add2_summary['ox_on_v_tangent']):
                        diff_t = add2_summary['ox_on_v_tangent'] - add1_summary['ox_on_v_tangent']

                    comparison_data.append({
                        '비교': f"**{add2_name}** vs **{add1_name}**",
                        'ΔV (미분법)': diff_d,
                        'ΔV (접선법)': diff_t
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df.style.format(
                    {'ΔV (미분법)': '{:+.3f} V', 'ΔV (접선법)': '{:+.3f} V'},
                    na_rep="N/A"
                ))
    
    # --- ### 상세 분석 데이터 (요청에 따라 제공된 코드 그대로 유지) ### ---
    st.subheader("📑 상세 분석 데이터")
    base_df_data = [{'시트 이름': name, '타입': s['type'], '산화 시작 (미분법)': s['ox_on_v_deriv'], '산화 시작 (접선법)': s['ox_on_v_tangent']} for name, s in analysis_summary.items()]
    display_df = pd.DataFrame(base_df_data)

    for ref_name in ref_sheets:
        ref_summary = analysis_summary.get(ref_name, {})
        ref_v_d, ref_v_t = ref_summary.get('ox_on_v_deriv'), ref_summary.get('ox_on_v_tangent')
        display_df[f"ΔV vs {ref_name} (미분법)"] = display_df.apply(lambda r: (r['산화 시작 (미분법)'] - ref_v_d) if r['타입']=='Additive' and pd.notna(r['산화 시작 (미분법)']) and pd.notna(ref_v_d) else np.nan, axis=1)
        display_df[f"ΔV vs {ref_name} (접선법)"] = display_df.apply(lambda r: (r['산화 시작 (접선법)'] - ref_v_t) if r['타입']=='Additive' and pd.notna(r['산화 시작 (접선법)']) and pd.notna(ref_v_t) else np.nan, axis=1)
    
    numeric_cols = [col for col in display_df.columns if '산화 시작' in col or 'ΔV' in col]
    st.dataframe(display_df.style.format("{:.3f}", na_rep="N/A", subset=numeric_cols))


    st.subheader("📁 결과 저장")
    df_for_report = display_df.fillna('N/A')
    report_text = f"## 상세 데이터 (분석 파라미터: {analysis_params}) ##\n{df_for_report.to_markdown(index=False)}"
    
    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button("📊 리포트 다운로드 (.txt)", report_text, f"{uploaded_file.name}_report.txt")
    try:
        if selected_sheets:
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            dl_col2.download_button("📈 그래프 다운로드 (.png)", img_bytes, f"{uploaded_file.name}_graph.png")
    except ValueError:
        dl_col2.warning("그래프 다운로드를 위해 Kaleido 패키지가 필요합니다. (pip install kaleido)")