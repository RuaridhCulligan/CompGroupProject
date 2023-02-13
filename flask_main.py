from flask import Flask, redirect, url_for, render_template, request, send_file, send_from_directory
import os
from NumericalMethods.main import main

log_file = os.path.join("Numerical Methods", "log.txt")
#pdf_output = os.path.join(main_dir_path, main_dir_path, "Numerical Methods", "output", "visualisation.pdf")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/caseA.html", methods = ["POST", "GET"])
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
        return caseAdisplay()

    return render_template("caseA.html")

@app.route("/caseB.html", methods = ["POST", "GET"])
def caseB():
    return render_template("caseB.html")


@app.route('/caseAdisplay/') #the url you'll send the user to when he wants the pdf
def caseAdisplay():
    return redirect("/static/docs/visualisation.pdf") #the pdf itself

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
        output.append("SHOW_V {0}\n".format(data[4]))
        output.append("SAVE 0\n")
        output.append("DIFF {0}\n".format(data[6]))
        #sys_par
        output.append("t_end {0}\n".format(data[7]))
        output.append("a {0}\n".format(data[8]))
        output.append("k0 {0}\n".format(data[9]))
        output.append("b {0}\n".format(data[10]))
        output.append("ky0 {0}\n".format(data[11]))
        output.append("V0 {0}\n".format(data[12]))
        output.append("d {0}\n".format(data[13]))
        output.append("w {0}\n".format(data[14]))
        output.append("alpha {0}\n".format(data[15]))
        output.append("x0 {0}\n".format(data[16]))
        output.append("y0 {0}\n".format(data[17]))
        #num_par
        output.append("x_min {0}\n".format(data[18]))
        output.append("x_max {0}\n".format(data[19]))
        output.append("dx {0}\n".format(data[20]))
        output.append("dt {0}\n".format(data[21]))
        output.append("y_min {0}\n".format(data[22]))
        output.append("y_max {0}\n".format(data[23]))
        output.append("dy {0}".format(data[24]))
    file = open(log_file, "w")
    file.writelines(output)
    file.close()


if __name__ == "__main__":
    app.run(debug = True)

