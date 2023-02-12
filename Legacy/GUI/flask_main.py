from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/caseA.html", methods = ["POST", "GET"])
def caseA():
    data = []
    if request.method == "POST":
        data.append(request.form["t1"])
        data.append(request.form["dt"])
        data.append(request.form["x0"])
        data.append(request.form["x1"])
        data.append(request.form["dx"])
        data.append(request.form["a"])
        data.append(request.form["k0"])
        data.append(request.form["int_method"])
        data.append(request.form["snapshot"])
        write_to_formdata("caseA", data)

    return render_template("caseA.html")

@app.route("/caseB.html", methods = ["POST", "GET"])
def caseB():
    return render_template("caseB.html")


def write_to_formdata(case, data):
    output = []
    if case == "caseA":
        output.append("case {0}\n".format(case))
        output.append("t1 {0}\n".format(data[0]))
        output.append("dt {0}\n".format(data[1]))
        output.append("x0 {0}\n".format(data[2]))
        output.append("x1 {0}\n".format(data[3]))
        output.append("dx {0}\n".format(data[4]))
        output.append("a {0}\n".format(data[5]))
        output.append("k0 {0}\n".format(data[6]))
        output.append("int_method {0}\n".format(data[7]))
        output.append("snapshot {0}".format(data[8]))
    file = open("log.txt", "w")
    file.writelines(output)


if __name__ == "__main__":
    app.run(debug = True)

