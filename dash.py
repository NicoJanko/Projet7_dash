
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import subprocess


st.set_page_config(page_title='Prêt à dépenser', layout = 'wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.initjs()
feat_des = pd.read_csv('feat_des.csv')

#request functions:
#unitary test
@st.cache_resource
def test():
    pytest_results = subprocess.run(['pytest', 'test_api.py'], capture_output=True)
    if pytest_results.returncode != 0:
        raise Exception('Tests failed : {}'.format(pytest_results.stdout.decode('utf-8')))
       


#prediction request
@st.cache_resource
def make_pred(api_uri, client_id):
    response = requests.get(api_uri+'/predict', json={'client_id' : client_id}
                               )
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()
 
def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html)

#individual feature request
def get_feat(api_uri, client_id, feat):
    response = requests.get(api_uri+'/feat', json={'id' : client_id, 'feat' : feat})
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

#summary plot request
@st.cache_resource
def get_summary(api_uri):
    response = requests.get(api_uri+'/summary')
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()



#text to prompt on the help bubbles
help_feat = 'Choisissez une variable pour afficher la valeur du client, ainsi que sa position dans la distribution'
help_predict = 'Cochez la case pour obtenir la prédiciton ainsi que les données explicatives du client.'
help_client = "Entrez l'identifiant du client."

#main app
def main():
    api_uri = 'http://pad-app.herokuapp.com'
    #unitary tests
    test()
    #load one-timed features
    summ_response = get_summary(api_uri)
    feat_name = feat_des['Row'].to_list()
    feat_name = sorted(feat_name)
    with st.sidebar:
        #selection tools on the sidebar
        st.title('Prêt à dépenser')
        client_selector = st.number_input("Identifiant client",
                                              min_value = 100001,
                                              max_value = 456250,
                                              help = help_client
                                             )
        predict_btn = st.checkbox('Prédire', help = help_predict)

        feature_selector = st.selectbox('Choisir variable :', feat_name, help=help_feat)
        

    if predict_btn:
        #main display
        response = make_pred(api_uri, client_selector)
        proba = response['probability']
        pred = response['prediction']
        if pred == 0:
            st.header(':blue[ACCEPTE]')
        else: st.header(':red[REFUSE]')
        col1, col2 = st.columns(2)
        with col1:
            st.header('Probabilité de non-remboursement :')
        with col2:
            if pred == 0 :
                st.header(':blue[{}]'.format(proba)+':blue[%]')
            else: st.header(':red[{}]'.format((proba))+':red[%]')
        #local shap plot    
        st_shap(shap.force_plot(response['expected_val'], np.array(response['shap_values']), pd.DataFrame(response['data'])))
        with st.expander("Plus d'information"):
            st.write(
                """Ce graphique aide à comprendre le calcul de la probabilité pour un client.
                En rouge (gauche) sont les variables ayant contribué le plus à une probabilité forte de non-remboursement.
                En bleu (droite) sont les variables ayant contribué le plus à un probabilité faible de non-remboursement.
                Si un client possède une probabilité forte de non remboursement, cela est dû aux variables rouges."""
                )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            #shap global summary plot
            st.subheader('Variables les plus influentes')
            shap.summary_plot(np.array(summ_response['rand_sv']),
                             pd.DataFrame(summ_response['rand_data']),
                             )
            st.pyplot(bbox_inches='tight')
            with st.expander("Plus d'information"):
                st.write(
                    """Ce graphique représente les variables les plus influantes pour le calcul de la probabilité.
                    Par exemple, une valeur forte de "AMT_REQ_CREDIT_BUREAU_QRT" participe à l'augmentation de la probabilité de non-remboursement."""
                    )
        
        with col2:
            #single feature explainer
            st.subheader('Variable du client')
            response2 = get_feat(api_uri, client_selector, feature_selector)
            feat_val = pd.DataFrame(response2['raw_data'])
            st.table(feat_val)
            with st.expander('Description de la variable'):
                description = feat_des[feat_des['Row'] == feat_val.columns[0]]['Description'].values[0]
                st.write(f'{description}')
        with col3:
            #single feature distribution
            st.subheader('Variable du client par rapport aux autres clients')
            col_values = pd.DataFrame(response2['col_values'])
            col_values = col_values.iloc[:,-1]
            fig, ax = plt.subplots()
            if col_values.dtype == 'float64':
                ax.hist(col_values)
                ax.axvline(feat_val.values,
                          color='red',
                         linestyle='dotted',
                         linewidth=2,
                         )
            else:
                ax.pie(col_values.value_counts(), labels = col_values.unique())
            st.pyplot(fig)
            with st.expander("Plus d'information"):
                st.write(
                    """Ce graphique vous permet de situer le client parmis l'ensemble des clients pour une variable donnée.
                    Dans le cas de l'histograme, le bar en rouge montre la position du client dans la distribution"""
                    )
           

    
    
if __name__ == '__main__':
    main()