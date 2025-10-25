import Circle from "./components/Circle"
import "./App.css"

function App() {

  return (
    <>
      <h6>Auto grad viz
        <span> - A tool for visualizing automatic differentiation</span>
      </h6>
      <div className="card">
        <Circle x={100} y={100} radius={40} color="blue" numberOfCircles={5} />
      </div>
    </>
  )
}

export default App
