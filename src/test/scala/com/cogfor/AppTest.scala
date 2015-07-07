package com.cogfor

import org.scalatest.FunSuite
 class AppTest extends FunSuite {
  
  test("list length") {
    assertResult(3) {
      App.listLength(List(1,2,3))
    }
  }
}
	
