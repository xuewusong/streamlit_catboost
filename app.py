import streamlit as st
import joblib
import matplotlib.pyplot as plt
from myheader import col # 导入模型每列类型
import shap
import pandas as pd

# 设置页面信息
st.set_page_config(page_title="基于CatBoost的老年缺血性心脏病患者再入院预测系统", layout="wide", initial_sidebar_state="auto")

# 导入模型
m = joblib.load('CatBoost.pkl')

# 导入训练数据
train_data = pd.read_csv('train_data.csv',header=0)
train_data = pd.DataFrame(train_data)
test_data = pd.read_csv('test_data.csv',header=0)
test_data = pd.DataFrame(test_data)

X_train = train_data.drop("1-year readmission",axis=1)
Y_train = train_data["1-year readmission"]

# 获取列名称
COL = X_train.columns.tolist()

# 默认值设置
default_value = {i:j for i, j in zip(COL, X_train.values[0])} # 设置默认输入

# 获取模型因素重要性
feature_importances = m.get_feature_importance()  
features = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])  
df0 = pd.DataFrame({"feature importances":feature_importances, "features":features})
df0 = df0.sort_values(by="feature importances") # 进行排序

BOOL = {"No":0, "Yes":1}
SEX = {"Female":0, "Male":1}
Educational_level = {
    "No education":0,
    "Primary school":1,
    "Middle school":2,
    "High school":3,
    "College or above":4
}
dw = {
    "LOS":"days",
    "DBP":"mmHg",
    "LA":"mm",
    "RA":"mm",
    "NOM":"types",
    "FT3":"pmol/L",
    "FT4":"pmol/L",
    "PTA":"%",
    "TT":"seconds",
    "Fibrinogen":"g/L",
    "LDLC":"mmol/L",
    "ApoB":"g/L",
    #"Albumin/globulin",
    "AST":"U/L",
    "Indirect bilirubin":"µmol/L",
    "Creatinine":"µmol/L",
    "Uric acid":"µmol/L",
    "LDH":"U/L",
    "CKMB":"µg/L",
    "Glucose":"mmol/L",
    "Monocyte":"*10^9/L",
    "Basophil":"*10^9/L",
    "RBC":"*10^12/L",
    "Hemoglobin":"g/L",
    "RDW-SD":"fL",
    "Serum sodium":"mmol/L"
}

# 设置图片样式
st.markdown("""
<style>
    [data-testid="stImage"] {
        border: 1px solid gray;
        border-radius: 0.5rem;
    }
    [data-testid="block-container"] {
        padding-top: 36px;
        padding-bottom: 36px;
    }
</style>""", unsafe_allow_html=True)

# 设置系统标题
st.markdown(f'''
        <div style="text-align: center; color: white; font-size: 24px; font-weight: bold; background: #ff4b4b; padding: 0.4rem; border-radius: .5rem; margin-bottom: 1rem;">
        基于CatBoost的老年缺血性心脏病患者再入院预测系统
        </div>''', unsafe_allow_html=True)

# 绘制模型因素重要性柱状图
def plot_importance():  
    feature_importances = m.get_feature_importance()  
    features = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])  
    df = pd.DataFrame({"feature importances":feature_importances, "features":features})
    df = df.sort_values(by="feature importances")
    df["features"] = df.apply(lambda x: x["features"]+f' ⬅ ({round(x["feature importances"], 2)})', axis=1)

    #fig = plt.figure(figsize=(8, 8), dpi=300)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300, facecolor="none")
    df.plot(kind="barh", y="feature importances", x="features", ax=ax, width=0.9, legend=False, color="#F57C00")
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    plt.gca().spines.bottom.set_visible(False)
    plt.grid(axis="x", ls="--", lw=0.5, alpha=0.5)
    plt.title("Feature importances", fontsize=14, fontweight="bold")
    
    st.pyplot(fig, use_container_width=True)

# 绘制预测瀑布图    
def plot_water_full():
    x = pd.DataFrame([st.session_state["data"]])
    x.columns = COL
    # 创建SHAP解释器  
    explainer = shap.Explainer(m)  
    shap_values = explainer(x)  

    # 绘制瀑布图  
    #plt.figure(figsize=(8, 4), dpi=300) 
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300, facecolor="none")    
    shap.waterfall_plot(shap_values[0], max_display=15)  
    
    st.pyplot(fig, use_container_width=True)

# 缓存，用来保存模型输入
if "data" not in st.session_state:
    st.session_state["data"] = {}

# 滑动条，用来通过因素重要性输入模型值
im = st.slider("特征重要性", value=(0., df0["feature importances"].max()), min_value=0., max_value=df0["feature importances"].max(), step=0.01)

# 模型输入部分
with st.form("input"):
    c = st.columns(6)
    k = 0
    for i in col: # 获取模型输入
        if i["name"] in df0[(df0["feature importances"]<= im[1]) & (df0["feature importances"]>= im[0])]["features"].tolist():
            if i["type"] == "choice":
                V = [int(j) for j in i['data']]
                V = sorted(V)
                if i["name"]=="Educational level":
                    st.session_state["data"][i["name"]] = Educational_level[c[k%6].selectbox(i["name"], list(Educational_level.keys()), index=i['data'].index(default_value[i["name"]]))]
                elif i["name"]=="Sex":
                    st.session_state["data"][i["name"]] = SEX[c[k%6].selectbox(i["name"], list(SEX.keys()), index=i['data'].index(default_value[i["name"]]))]
                elif sum(V)==1:
                    st.session_state["data"][i["name"]] = BOOL[c[k%6].selectbox(i["name"], list(BOOL.keys()), index=i['data'].index(default_value[i["name"]]))]
                else:
                    st.session_state["data"][i["name"]] = c[k%6].selectbox(i["name"], V, index=i['data'].index(default_value[i["name"]]))
            else:
                if i["name"] in dw:
                    NAME = i["name"]+f'({dw[i["name"]]})'
                else:
                    NAME = i["name"]
                if i["dtype"]=="int":
                    st.session_state["data"][i["name"]] = c[k%6].number_input(NAME, step=i["step"], value=int(default_value[i["name"]]))
                else:
                    st.session_state["data"][i["name"]] = c[k%6].number_input(NAME, step=i["step"], value=default_value[i["name"]])
            k = k+1
    
    c = st.columns(5)
    submit = c[2].form_submit_button("开始预测", use_container_width=True) # 预测按钮

# 预测按钮提交
if submit:
    x = pd.DataFrame([st.session_state["data"]])
    x.columns = COL
    
    D = {0:"否", 1:"是"}
    res = m.predict(x) # 进行预测
    res_p = {i:round(j, 2) for i, j in zip([0, 1], m.predict_proba(x)[0])} # 预测概率
    
    # 如果点击了预测按钮则隐藏模型输入
    st.markdown("""
        <style>
            [data-testid="stForm"],
            .stSlider {
               display: none; 
            }
        </style>""", unsafe_allow_html=True)
    
    # 展示预测输入
    with st.expander("**🔸当前预测输入值**", False):
            st.dataframe(x, hide_index=True, use_container_width=True)
    # 展示预测结果
    with st.expander("**〽️预测结果可视化**", True):
        st.markdown(f'''
        <div style="text-align: center; color: red; font-size:20px; font-weight: bold; padding: 1rem; margin-bottom: 0.5rem; border-bottom: 1px solid black;">
        当前预测结果为：{int(res[0])} ⬅ {res_p}, 是否会造成老年缺血性心脏病患者再入院？（{D[int(res)]}）
        </div>''', unsafe_allow_html=True)
        c = st.columns([1,4,4,1])
        
        with c[1]:
            plot_importance()
        with c[2]:
            plot_water_full()  
    
    c = st.columns(5)
    bt = c[2].button("🔄返回预测", use_container_width=True) # 返回重新预测
    
    if bt: # 如果点击返回预测，则显示模型输入部分
        st.markdown("""
        <style>
            [data-testid="stForm"],
            .stSlider {
               display: show; 
            }
        </style>""", unsafe_allow_html=True)
    
