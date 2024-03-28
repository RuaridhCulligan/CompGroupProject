from flask import Flask, redirect, url_for, render_template, request, send_file, send_from_directory
import os
from NumericalMethods.main import main

log_file = os.path.join("NumericalMethods", "log.txt")



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/caseA", methods = ["POST", "GET"])
def caseA():
    data = []
    if request.method == "POST":
        data.append(request.form["METHOD"])
        data.append(request.form["STATIC"])
        #settings
        data.append(request.form["HIDE_A"])
        data.append(request.form["ADD_MET"])
        data.append(request.form["DIFF"])
        #sys_par
        data.append(request.form["t_end"])
        data.append(request.form["a"])
        data.append(request.form["k0"])
        data.append(request.form["x0"])
        #num_par
        data.append(request.form["x_min"])
        data.append(request.form["x_max"])
        data.append(request.form["dx"])
        data.append(request.form["dt"])
        write_to_formdata("caseA", data)
        main()
        if data[1] == "1.0" or data[1] == "0.5":
            return one_dim_display(False)
        elif data[1] == "0.0":
            return one_dim_display(True)


    return render_template("caseA.html")

@app.route("/caseB", methods = ["POST", "GET"])
def caseB():
    data = []
    if request.method == "POST":
        data.append(request.form["METHOD"])
        data.append(request.form["STATIC"])
        #settings
        data.append(request.form["HIDE_A"])
        data.append(request.form["ADD_MET"])
        data.append(request.form["DIFF"])
        #sys_par
        data.append(request.form["t_end"])
        data.append(request.form["a"])
        data.append(request.form["k0"])
        data.append(request.form["x0"])
        data.append(request.form["ky0"])
        #num_par
        data.append(request.form["x_min"])
        data.append(request.form["x_max"])
        data.append(request.form["dx"])
        data.append(request.form["dt"])
        write_to_formdata("caseB", data)
        main()
        if data[1] == "1.0" or data[1] == "0.5":
            return two_dim_display(False)
        elif data[1] == "0.0":
            return two_dim_display(True)
    return render_template("caseB.html")

@app.route("/caseC", methods = ["POST", "GET"])
def caseC():
    data = []
    if request.method == "POST":
        data.append(request.form["METHOD"])
        data.append(request.form["STATIC"])
        #settings
        data.append(request.form["ADD_MET"])
        data.append(request.form["SHOW_V"])
        #sys_par
        data.append(request.form["t_end"])
        data.append(request.form["a"])
        data.append(request.form["k0"])
        data.append(request.form["V0"])
        data.append(request.form["d"])
        data.append(request.form["x0"])
        #num_par
        data.append(request.form["x_min"])
        data.append(request.form["x_max"])
        data.append(request.form["dx"])
        data.append(request.form["dt"])
        write_to_formdata("caseC", data)
        main()
        if data[1] == "1.0" or data[1] == "0.5":
            return one_dim_display(False)
        elif data[1] == "0.0":
            return one_dim_display(True)
    return render_template("caseC.html")

@app.route("/caseD", methods = ["POST", "GET"])
def caseD():
    data = []
    if request.method == "POST":
        data.append(request.form["METHOD"])
        data.append(request.form["STATIC"])
        #settings
        data.append(request.form["ADD_MET"])
        data.append(request.form["SHOW_V"])
        #sys_par
        data.append(request.form["t_end"])
        data.append(request.form["a"])
        data.append(request.form["b"])
        data.append(request.form["k0"])
        data.append(request.form["ky0"])
        data.append(request.form["x0"])
        #num_par
        data.append(request.form["x_min"])
        data.append(request.form["x_max"])
        data.append(request.form["dx"])
        data.append(request.form["dt"])
        #
        data.append(request.form["y0"])
        data.append(request.form["d"])
        data.append(request.form["w"])
        write_to_formdata("caseD", data)
        main()
        if data[1] == "1.0" or data[1] == "0.5":
            return two_dim_display(False)
        elif data[1] == "0.0":
            return two_dim_display(True)
    return render_template("caseD.html")

@app.route("/caseE", methods = ["POST", "GET"])
def caseE():
    data = []
    if request.method == "POST":
        data.append(request.form["METHOD"])
        data.append(request.form["STATIC"])
        #settings
        data.append(request.form["ADD_MET"])
        #sys_par
        data.append(request.form["t_end"])
        data.append(request.form["a"])
        data.append(request.form["b"])
        data.append(request.form["k0"])
        data.append(request.form["ky0"])
        data.append(request.form["x0"])
        data.append(request.form["y0"])
        #num_par
        data.append(request.form["x_min"])
        data.append(request.form["x_max"])
        data.append(request.form["dx"])
        data.append(request.form["dt"])
        #
        data.append(request.form["alpha"])
        write_to_formdata("caseE", data)
        main()
        if data[1] == "1.0" or data[1] == "0.5":
            return one_dim_display(False)
        elif data[1] == "0.0":
            return one_dim_display(True)
    return render_template("caseE.html")

#displaying PDFs/GIFs for one dim or two dim cases
@app.route('/display')
def one_dim_display(gif):
    if gif == False:
        return redirect("/static/docsoutput/visualisation.pdf")
    else:
        return redirect("/static/docsoutput/visualisation.gif")

@app.route('/display')
def two_dim_display(gif):
    if gif == False:
        return redirect("/static/docsoutput/visualisation2D.pdf")
    else:
        return redirect("/static/docsoutput/visualisation2D.gif")

def write_to_formdata(case, data):
    output = []
    if case == "caseA":
        #independent
        output.append("CASE {0}\n".format(case))
        output.append("METHOD {0}\n".format(data[0]))
        output.append("STATIC {0}\n".format(data[1]))
        #settings
        output.append("HIDE_A {0}\n".format(data[2]))
        output.append("ADD_MET {0}\n".format(data[3]))
        output.append("SHOW_V 0\n")
        output.append("SAVE 0\n")
        output.append("DIFF {0}\n".format(data[4]))
        #sys_par
        output.append("t_end {0}\n".format(data[5]))
        output.append("a {0}\n".format(data[6]))
        output.append("k0 {0}\n".format(data[7]))
        output.append("b 1\n")
        output.append("ky0 1\n")
        output.append("V0 100\n")
        output.append("d 10\n")
        output.append("w 2\n")
        output.append("alpha 0.5\n")
        output.append("x0 {0}\n".format(data[8]))
        output.append("y0 -5\n")
        #num_par
        output.append("x_min {0}\n".format(data[9]))
        output.append("x_max {0}\n".format(data[10]))
        output.append("dx {0}\n".format(data[11]))
        output.append("dt {0}\n".format(data[12]))
        output.append("y_min -10\n")
        output.append("y_max 10\n")
        output.append("dy 0.1")
    elif case == "caseB":
        #independent
        output.append("CASE {0}\n".format(case))
        output.append("METHOD {0}\n".format(data[0]))
        output.append("STATIC {0}\n".format(data[1]))
        #settings
        output.append("HIDE_A {0}\n".format(data[2]))
        output.append("ADD_MET {0}\n".format(data[3]))
        output.append("SHOW_V 0\n")
        output.append("SAVE 0\n")
        output.append("DIFF {0}\n".format(data[4]))
        #sys_par
        output.append("t_end {0}\n".format(data[5]))
        output.append("a {0}\n".format(data[6]))
        output.append("k0 {0}\n".format(data[7]))
        output.append("b 1\n")
        output.append("ky0 {0}\n".format(data[9]))
        output.append("V0 0\n")
        output.append("d 0\n")
        output.append("w 0\n")
        output.append("alpha 0\n")
        output.append("x0 {0}\n".format(data[8]))
        output.append("y0 {0}\n".format(data[8]))
        #num_par
        output.append("x_min {0}\n".format(data[10]))
        output.append("x_max {0}\n".format(data[11]))
        output.append("dx {0}\n".format(data[12]))
        output.append("dt {0}\n".format(data[13]))
        output.append("y_min {0}\n".format(data[10]))
        output.append("y_max {0}\n".format(data[11]))
        output.append("dy {0}".format(data[12]))
    elif case == "caseC":
            #independent
        output.append("CASE {0}\n".format(case))
        output.append("METHOD {0}\n".format(data[0]))
        output.append("STATIC {0}\n".format(data[1]))
        #settings
        output.append("HIDE_A 1\n")
        output.append("ADD_MET {0}\n".format(data[2]))
        output.append("SHOW_V {0}\n".format(data[3]))
        output.append("SAVE 0\n")
        output.append("DIFF 0\n")
        #sys_par
        output.append("t_end {0}\n".format(data[4]))
        output.append("a {0}\n".format(data[5]))
        output.append("k0 {0}\n".format(data[6]))
        output.append("b 1\n")
        output.append("ky0 0\n")
        output.append("V0 {0}\n".format(data[7]))
        output.append("d {0}\n".format(data[8]))
        output.append("w 0\n")
        output.append("alpha 0\n")
        output.append("x0 {0}\n".format(data[9]))
        output.append("y0 0\n")
        #num_par
        output.append("x_min {0}\n".format(data[10]))
        output.append("x_max {0}\n".format(data[11]))
        output.append("dx {0}\n".format(data[12]))
        output.append("dt {0}\n".format(data[13]))
        output.append("y_min 0\n")
        output.append("y_max 0\n")
        output.append("dy 0")
    elif case == "caseD":
        #independent
        output.append("CASE {0}\n".format(case))
        output.append("METHOD {0}\n".format(data[0]))
        output.append("STATIC {0}\n".format(data[1]))
        #settings
        output.append("HIDE_A 1\n")
        output.append("ADD_MET {0}\n".format(data[2]))
        output.append("SHOW_V {0}\n".format(data[3]))
        output.append("SAVE 0\n")
        output.append("DIFF 0\n")
        #sys_par
        output.append("t_end {0}\n".format(data[4]))
        output.append("a {0}\n".format(data[5]))
        output.append("k0 {0}\n".format(data[7]))
        output.append("b {0}\n".format(data[6]))
        output.append("ky0 {0}\n".format(data[8]))
        output.append("V0 0\n")
        output.append("d {0}\n".format(data[15]))
        output.append("w {0}\n".format(data[16]))
        output.append("alpha 0\n")
        output.append("x0 {0}\n".format(data[9]))
        output.append("y0 {0}\n".format(data[14]))
        #num_par
        output.append("x_min {0}\n".format(data[10]))
        output.append("x_max {0}\n".format(data[11]))
        output.append("dx {0}\n".format(data[12]))
        output.append("dt {0}\n".format(data[13]))
        output.append("y_min {0}\n".format(data[10]))
        output.append("y_max {0}\n".format(data[11]))
        output.append("dy {0}".format(data[12]))
    elif case == "caseE":
        #independent
        output.append("CASE {0}\n".format(case))
        output.append("METHOD {0}\n".format(data[0]))
        output.append("STATIC {0}\n".format(data[1]))
        #settings
        output.append("HIDE_A 1\n")
        output.append("ADD_MET {0}\n".format(data[2]))
        output.append("SHOW_V 0\n")
        output.append("SAVE 0\n")
        output.append("DIFF 0\n")
        #sys_par
        output.append("t_end {0}\n".format(data[3]))
        output.append("a {0}\n".format(data[4]))
        output.append("k0 {0}\n".format(data[6]))
        output.append("b {0}\n".format(data[5]))
        output.append("ky0 {0}\n".format(data[7]))
        output.append("V0 0\n")
        output.append("d 0\n")
        output.append("w 0\n")
        output.append("alpha {0}\n".format(data[14]))
        output.append("x0 {0}\n".format(data[8]))
        output.append("y0 {0}\n".format(data[9]))
        #num_par
        output.append("x_min {0}\n".format(data[10]))
        output.append("x_max {0}\n".format(data[11]))
        output.append("dx {0}\n".format(data[12]))
        output.append("dt {0}\n".format(data[13]))
        output.append("y_min {0}\n".format(data[10]))
        output.append("y_max {0}\n".format(data[11]))
        output.append("dy {0}".format(data[12]))
    file = open(log_file, "w")
    file.writelines(output)
    file.close()


if __name__ == "__main__":
    app.run(debug = True)

