from flask import Flask, send_file, render_template, request, redirect
from io import BytesIO
import seaborn as sns
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import solvers
import levenberg


app = Flask(__name__)

def get_y_pred():
    global Y_pred, k, tau
    Y_pred = k*(1 - np.exp(-X/tau))


def get_xy():
    global npoints
    if npoints != -1:
        idxs = sorted(np.random.choice(solvers.X.shape[0], npoints, replace=False))
        myX = solvers.X[idxs]
        myY = solvers.y[idxs]
    else:
        npoints = len(solvers.X)
        myX = solvers.X
        myY = solvers.y
    return myX, myY


X = np.arange(0, 10, 0.1)
k = 0.5
tau = 1
npoints = len(solvers.X)
Y_pred = []
get_y_pred()
method = ""
cvg_hst = []


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/image')
def image():
    if method != "lev":
        # Step 1: Generate Matplotlib Figure
        fig, ax = plt.subplots()
        ax.scatter(solvers.X, solvers.y, marker='o')
        ax.set_xlabel('t')
        ax.set_ylabel('y(t)')
        ax.plot(X, Y_pred)
        ax.set_title(f"K={k:.2f}, tau={tau:.2f}, Pontos={npoints}, Método: {method.title()}")
    else:
        x = X
        y = Y_pred
        # extract parameters data
        p_hst  = cvg_hst[:,2:]
        p_fit  = p_hst[-1,:]
        y_fit = levenberg.lm_func(x,np.array([p_fit]).T)
    
        # define colors and markers used for plotting
        n = len(p_fit)
        colors = pl.cm.ocean(np.linspace(0,.75,n))
        markers = ['o','s','D','v']    
    
        # create plot of raw data and fitted curve
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.scatter(solvers.X,solvers.y,marker='o')
        ax1.plot(x,y_fit)
        ax1.set_xlabel('t')
        ax1.set_ylabel('y(t)')
        ax1.set_title(f"K={k:.2f}, tau={tau:.2f}, Pontos={npoints}, Método: {method.title()}")
    
        ax2 = fig.add_subplot(gs[1, 0])
        # create plot showing convergence of parameters
        for i in range(n):
            ax2.plot(cvg_hst[:,0],p_hst[:,i]/p_hst[0,i],color=colors[i],marker=markers[i],
                 linestyle='-',markeredgecolor='black',label='p'+'${_%i}$'%(i+1))
        ax2.set_xlabel('Iterações')
        ax2.set_ylabel('Valores (norma)')
        ax2.set_title('Convergencia de parametros') 
        ax2.legend()
            
        ax3 = fig.add_subplot(gs[1, 1])
        # create plot showing histogram of residuals
        sns.histplot(ax=ax3,data=y_fit-y,color='deepskyblue')
        ax3.set_xlabel('Erro residual')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Histograma de residuos')
        plt.tight_layout()

    # Step 2: Save Figure to BytesIO
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)

    # Step 3: Serve BytesIO Object in Flask
    return send_file(img_bytesio, mimetype='image/png')


@app.route('/gradient-image', methods=["POST"])
def gradient_image():
    global Y_pred, tau, k, npoints, method
    method = "grad"
    # Get form data
    npoints = int(request.form['npoints'])
    iterations = int(request.form['iterations'])
    learning_rate = float(request.form['learning_rate'])
    tolerance = float(request.form['tolerance'])
    myX, myY = get_xy()

    k, tau = solvers.gradient_descent(myX, myY, iterations, learning_rate, tolerance)
    Y_pred = k*(1 - np.exp(-X/tau))
    return redirect('/')


@app.route('/newton-image', methods=["POST"])
def newton_image():
    global Y_pred, tau, k, npoints, method
    method = "newton"
    # Get form data
    npoints = int(request.form['npoints'])
    iterations = int(request.form['iterations'])
    tolerance = float(request.form['tolerance'])
    k0 = float(request.form['k0'])
    tau0 = float(request.form['tau0'])
    
    myX, myY = get_xy()

    k, tau = solvers.Gauss_Newton(myX, myY, k0, tau0, tolerance, iterations)
    
    Y_pred = k*(1 - np.exp(-X/tau))
    return redirect('/')


@app.route('/levenberg-image', methods=["POST"])
def levenberg_image():
    global Y_pred, tau, k, npoints, method, cvg_hst
    method = "lev"
    # Get form data
    npoints = int(request.form['npoints'])
    k0 = float(request.form['k0'])
    tau0 = float(request.form['tau0'])
    
    myX, myY = get_xy()

    p_fit,Chi_sq,sigma_p,sigma_y,corr,R_sq,cvg_hst = levenberg.lm(
        np.array([[k0, tau0]]).T, myX, myY)
    k, tau = p_fit[0][0], p_fit[1][0]
    
    Y_pred = k*(1 - np.exp(-X/tau))
    return redirect('/')


@app.route('/bfgs-image', methods=["POST"])
def bfgs_image():
    global Y_pred, tau, k, npoints, method
    method = "bfgs"
    # Get form data
    npoints = int(request.form['npoints'])
    k0 = float(request.form['k0'])
    tau0 = float(request.form['tau0']) 
    iterations = int(request.form['iterations'])

    myX, myY = get_xy()

    k, tau = solvers.BFGS([k0, tau0], myX, myY, iterations)
    
    Y_pred = k*(1 - np.exp(-X/tau))
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
